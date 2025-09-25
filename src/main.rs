use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Write};
use std::fs;
use std::io::Write as _;
use std::io::stdout;
use std::iter::FusedIterator;
use std::ops::Index;
use std::str::FromStr;
use std::time::Instant;

fn main() {
    let mut words: Vec<Word> = load_word_list("words.txt");
    words.sort();

    let mut zero_words = Vec::new();
    for &word in &words {
        if zero_words
            .iter()
            .copied()
            .all(|w: Word| (w.mask & word.mask) != w.mask)
        {
            zero_words.push(word);
        }
    }

    let args: Vec<_> = std::env::args().skip(1).collect();

    let mut i = 0;
    while let Some(arg) = args.get(i) {
        i += 1;

        if arg == "-j" {
            rayon::ThreadPoolBuilder::new()
                .num_threads(args.get(i).unwrap().parse().unwrap())
                .build_global()
                .unwrap();

            i += 1;
        } else {
            solve_single_word(&arg, &words, &zero_words);
            return;
        }
    }

    solve_all(&words, &zero_words);
}

fn solve_single_word(answer: &str, words: &[Word], zero_words: &[Word]) {
    let answer = Word::from_str(answer).unwrap();
    println!("Solving {answer} using {} words...", words.len());

    let solution = solve_word(answer, words, zero_words, false, true);
    print_solution(solution);
}

fn load_word_list(path: &str) -> Vec<Word> {
    fs::read_to_string(path)
        .unwrap()
        .split('\n')
        .filter(|&s| !s.is_empty())
        .map(FromStr::from_str)
        .filter_map(Result::ok)
        .collect()
}

fn solve_all(words: &[Word], zero_words: &[Word]) {
    let mut answers = load_word_list("answers.txt");
    println!(
        "Solving {} answers using {} words...",
        answers.len(),
        words.len()
    );

    let (answer_tx, answer_rx) = crossbeam_channel::unbounded();
    let (solution_tx, solution_rx) = crossbeam_channel::unbounded();

    std::thread::spawn({
        let answers = answers.clone();
        move || {
            for answer in answers {
                answer_tx.send(answer).unwrap();
            }
        }
    });

    rayon::spawn_broadcast({
        let solution_tx = solution_tx.clone();
        let words = Vec::from(words);
        let zero_words = Vec::from(zero_words);
        move |_| {
            while let Ok(answer) = answer_rx.recv() {
                solution_tx
                    .send(solve_word(answer, &words, &zero_words, true, false))
                    .unwrap();
            }
        }
    });
    drop(solution_tx);

    let mut received = Vec::new();
    while let Ok(solution) = solution_rx.recv() {
        received.push(solution);

        while let Some(i) = received
            .iter()
            .position(|s: &DoneSolution| s.word == answers[0])
        {
            let solution = received.remove(i);
            answers.remove(0);

            print_solution(solution);
        }
    }

    if !received.is_empty() {
        println!("ERROR: not all solutions printed yet");
        received.sort_by_key(|s| s.word.letters);
        for solution in received {
            print_solution(solution);
        }
    }
}

fn solve_word(
    answer: Word,
    words: &[Word],
    zero_words: &[Word],
    limited: bool,
    logs: bool,
) -> DoneSolution {
    let start = Instant::now();
    let zero_solution = find_zero_solution(answer, zero_words);
    let seconds_zero = start.elapsed().as_secs_f32();
    if logs {
        println!("Searching for all grey solution...");
    }
    if let Some(solution) = zero_solution {
        return DoneSolution {
            kind: SolutionKind::Done,
            word: answer,
            score: 0,
            guesses: solution.guesses,
            seconds: start.elapsed().as_secs_f32(),
            seconds_zero,
            seconds_greedy: 0.,
            seconds_yellow: 0.,
            seconds_full: 0.,
        };
    }

    if logs {
        println!("Searching for greedy solution...");
    }
    let solution = find_greedy_solution(answer, words).unwrap_or(Solution {
        scores_at: [i32::MAX; 6],
        state: State::new(answer),
    });
    let seconds_greedy = start.elapsed().as_secs_f32() - seconds_zero;
    if logs && solution.scores_at[5] != i32::MAX {
        println!("Found greedy solution with {} score", solution.scores_at[5]);
    }

    if solution.scores_at[5] == 1 {
        return DoneSolution {
            kind: SolutionKind::Done,
            word: answer,
            score: solution.scores_at[5],
            guesses: solution.state.guesses,
            seconds: start.elapsed().as_secs_f32(),
            seconds_zero,
            seconds_greedy,
            seconds_yellow: 0.,
            seconds_full: 0.,
        };
    }

    if logs {
        println!("Searching for yellow solution...");
    }
    let solution = if limited {
        find_min_yellow_solution_limit(answer, words, solution)
    } else {
        Ok(find_min_yellow_solution(answer, words, solution))
    };
    let seconds_yellow = start.elapsed().as_secs_f32() - seconds_zero - seconds_greedy;
    let solution = match solution {
        Ok(solution) | Err(solution) if solution.score >= 100 => {
            let mut state = State::new(answer);
            state.guesses = solution.state.guesses;
            Solution {
                scores_at: [solution.score; _],
                state,
            }
        }

        Ok(solution) => {
            return DoneSolution {
                kind: SolutionKind::Done,
                word: answer,
                score: solution.score,
                guesses: solution.state.guesses,
                seconds: start.elapsed().as_secs_f32(),
                seconds_zero,
                seconds_greedy,
                seconds_yellow,
                seconds_full: 0.,
            };
        }

        Err(solution) => {
            return DoneSolution {
                kind: SolutionKind::Limited,
                word: answer,
                score: solution.score,
                guesses: solution.state.guesses,
                seconds: start.elapsed().as_secs_f32(),
                seconds_zero,
                seconds_greedy,
                seconds_yellow,
                seconds_full: 0.,
            };
        }
    };

    if logs {
        println!("Searching for full solution...");
    }
    let solution = if limited {
        find_min_solution_limit(answer, words, solution)
    } else {
        Ok(find_min_solution(answer, words, solution))
    };
    let seconds_full =
        start.elapsed().as_secs_f32() - seconds_zero - seconds_greedy - seconds_yellow;
    match solution {
        Ok(solution) | Err(solution) if solution.score == i32::MAX => DoneSolution {
            kind: SolutionKind::Impossible,
            word: answer,
            score: solution.score,
            guesses: solution.state.guesses,
            seconds: start.elapsed().as_secs_f32(),
            seconds_zero,
            seconds_greedy,
            seconds_yellow,
            seconds_full,
        },

        Ok(solution) => DoneSolution {
            kind: SolutionKind::Done,
            word: answer,
            score: solution.score,
            guesses: solution.state.guesses,
            seconds: start.elapsed().as_secs_f32(),
            seconds_zero,
            seconds_greedy,
            seconds_yellow,
            seconds_full,
        },

        Err(solution) => DoneSolution {
            kind: SolutionKind::Limited,
            word: answer,
            score: solution.score,
            guesses: solution.state.guesses,
            seconds: start.elapsed().as_secs_f32(),
            seconds_zero,
            seconds_greedy,
            seconds_yellow,
            seconds_full,
        },
    }
}

struct DoneSolution {
    kind: SolutionKind,
    word: Word,
    score: i32,
    guesses: GuessVec,
    seconds: f32,
    seconds_zero: f32,
    seconds_greedy: f32,
    seconds_yellow: f32,
    seconds_full: f32,
}

enum SolutionKind {
    Done,
    Limited,
    Impossible,
}

fn print_solution(solution: DoneSolution) {
    match solution.kind {
        SolutionKind::Done => {
            let mut out = stdout().lock();
            _ = writeln!(
                out,
                "{} ({}) ({:.3}s from {:.3}, {:.3}, {:.3}, {:.3})",
                solution.word,
                solution.score,
                solution.seconds,
                solution.seconds_zero,
                solution.seconds_greedy,
                solution.seconds_yellow,
                solution.seconds_full,
            );
            _ = writeln!(out, "-----");
            for guess in solution.guesses {
                _ = writeln!(out, "{guess}");
            }
            _ = writeln!(out,);
        }
        SolutionKind::Limited => {
            let mut out = stdout().lock();
            _ = writeln!(
                out,
                "{} ({}) (limited) ({:.3}s from {:.3}, {:.3}, {:.3}, {:.3})",
                solution.word,
                solution.score,
                solution.seconds,
                solution.seconds_zero,
                solution.seconds_greedy,
                solution.seconds_yellow,
                solution.seconds_full,
            );
            _ = writeln!(out, "-----");
            for guess in solution.guesses {
                _ = writeln!(out, "{guess}");
            }
            _ = writeln!(out,);
        }
        SolutionKind::Impossible => {
            let mut out = stdout().lock();
            _ = writeln!(
                out,
                "{} (IMPOSSIBLE) ({:.3}s from {:.3}, {:.3}, {:.3}, {:.3})",
                solution.word,
                solution.seconds,
                solution.seconds_zero,
                solution.seconds_greedy,
                solution.seconds_yellow,
                solution.seconds_full,
            );
            _ = writeln!(out,);
        }
    }
}

fn find_greedy_solution(answer: Word, words: &[Word]) -> Option<Solution> {
    let mut scores_at = [0; 6];
    let mut state = State::new(answer);

    for depth in 0..6 {
        let mut min_solution = None::<(i32, Word)>;

        for &guess in words {
            if !state.valid_guess(guess) {
                continue;
            }

            let guess_score;
            if let Some((score, _)) = min_solution {
                if score <= state.min_score(guess) {
                    continue;
                }

                guess_score = state.eval_score(guess);
                if score <= guess_score {
                    continue;
                }
            } else {
                guess_score = state.eval_score(guess);
            }

            min_solution = Some((guess_score, guess));
        }

        let Some((guess_score, word)) = min_solution else {
            return None;
        };

        state.apply(word);
        if depth > 0 {
            scores_at[depth] = scores_at[depth - 1];
        }
        scores_at[depth] += guess_score;
        // println!("{word} (+{guess_score}/{})", scores_at[depth]);
    }

    Some(Solution { scores_at, state })
}

fn find_zero_solution(answer: Word, words: &[Word]) -> Option<ZeroState> {
    let words: Vec<_> = words
        .iter()
        .copied()
        .rev()
        .filter(|&w| (w.mask & answer.mask) == 0)
        .collect();

    let mut stack = Vec::new();
    stack.push(PartialSolutionZero {
        next_index: 0,
        state: ZeroState::new(answer),
    });

    'outer: while let Some(partial) = stack.last_mut() {
        while let Some(&guess) = words.get(partial.next_index as usize) {
            partial.next_index += 1;

            if !partial.state.valid_guess(guess) {
                continue;
            }

            let mut child_state = partial.state;
            child_state.apply(guess);

            let child_partial = PartialSolutionZero {
                next_index: partial.next_index,
                state: child_state,
            };

            if child_partial.state.guesses.len() == 6 {
                return Some(child_partial.state);
            } else {
                stack.push(child_partial);
                continue 'outer;
            }
        }

        stack.pop();
    }

    None
}

fn find_min_solution_limit(
    answer: Word,
    words: &[Word],
    better_than: Solution,
) -> Result<SolutionSingle, SolutionSingle> {
    let mut stack = Vec::new();
    stack.push(PartialSolutionSingle {
        next_index: 0,
        score: 0,
        state: State::new(answer),
    });

    let mut processed = 0usize;
    let mut min_solution = SolutionSingle {
        score: better_than.scores_at[5],
        state: better_than.state,
    };

    'outer: while let Some(partial) = stack.last_mut() {
        let depth = partial.state.guesses.len() as i32;
        while let Some(&guess) = words.get(partial.next_index as usize) {
            processed += 1;
            if processed % 10_000_000_000 == 0 {
                return Err(min_solution);
            }

            partial.next_index += 1;

            if !partial.state.valid_guess(guess) {
                continue;
            }

            if min_solution.score < partial.score + partial.state.min_score(guess) * (6 - depth) {
                continue;
            }

            let word_score = partial.state.eval_score(guess);
            if min_solution.score < partial.score + word_score * (6 - depth) {
                continue;
            }
            let new_score = partial.score + word_score;

            if partial.state.guesses.len() == 5 && new_score == min_solution.score {
                continue;
            }

            let mut child_state = partial.state.clone();
            child_state.apply(guess);

            let child_partial = PartialSolutionSingle {
                score: new_score,
                next_index: 0,
                state: child_state,
            };

            if child_partial.state.guesses.len() == 6 {
                min_solution = SolutionSingle {
                    score: child_partial.score,
                    state: child_partial.state,
                };

                if new_score <= 1 {
                    return Ok(min_solution);
                }

                continue;
            } else {
                stack.push(child_partial);
                continue 'outer;
            }
        }

        stack.pop();
    }

    Ok(min_solution)
}

fn find_min_yellow_solution_limit(
    answer: Word,
    words: &[Word],
    better_than: Solution,
) -> Result<SolutionYellow, SolutionYellow> {
    // filter out any words with greens
    let words: Vec<_> = words
        .iter()
        .copied()
        .filter(|&w| {
            w.letters
                .into_iter()
                .zip(answer.letters)
                .all(|(a, b)| a != b)
        })
        .collect();
    let mut stack = Vec::new();

    stack.push(PartialSolutionYellow {
        next_index: 0,
        score: 0,
        state: YellowState::new(answer),
    });

    let mut processed = 0usize;
    let mut min_solution = SolutionYellow {
        score: better_than.scores_at[5],
        state: {
            let mut state = YellowState::new(better_than.state.word);
            state.guesses = better_than.state.guesses;
            state
        },
    };

    'outer: while let Some(partial) = stack.last_mut() {
        let depth = partial.state.guesses.len() as i32;

        while let Some(&guess) = words.get(partial.next_index as usize) {
            processed += 1;
            if processed % 10_000_000_000 == 0 {
                return Err(min_solution);
            }

            partial.next_index += 1;

            if !partial.state.valid_guess(guess) {
                continue;
            }

            if min_solution.score < partial.score + partial.state.min_score(guess) * (6 - depth) {
                continue;
            }

            let word_score = partial.state.eval_score(guess);
            if min_solution.score < partial.score + word_score * (6 - depth) {
                continue;
            }

            let new_score = partial.score + word_score;
            if partial.state.guesses.len() == 5 && new_score == min_solution.score {
                continue;
            }

            let mut child_state = partial.state.clone();
            child_state.apply(guess);

            let child_partial = PartialSolutionYellow {
                score: new_score,
                next_index: 0,
                state: child_state,
            };

            if child_partial.state.guesses.len() == 6 {
                min_solution = SolutionYellow {
                    score: child_partial.score,
                    state: child_partial.state,
                };

                // see same comment in find_min_min_solution
                if new_score <= 1 {
                    return Ok(min_solution);
                }

                continue;
            } else {
                stack.push(child_partial);
                continue 'outer;
            }
        }

        stack.pop();
    }

    Ok(min_solution)
}

fn find_min_yellow_solution(answer: Word, words: &[Word], better_than: Solution) -> SolutionYellow {
    let start = Instant::now();

    // filter out any words with greens
    let words: Vec<_> = words
        .iter()
        .copied()
        .filter(|&w| {
            w.letters
                .into_iter()
                .zip(answer.letters)
                .all(|(a, b)| a != b)
        })
        .collect();

    let mut stack = Vec::new();
    stack.push(PartialSolutionYellow {
        next_index: 0,
        score: 0,
        state: YellowState::new(answer),
    });

    let mut processed = 0usize;
    let mut min_solution = SolutionYellow {
        score: better_than.scores_at[5],
        state: {
            let mut state = YellowState::new(better_than.state.word);
            state.guesses = better_than.state.guesses;
            state
        },
    };

    let mut root_index = 0;

    'outer: while let Some(partial) = stack.last_mut() {
        let depth = partial.state.guesses.len() as i32;
        let outermost = depth == 0;

        while let Some(&guess) = words.get(partial.next_index as usize) {
            processed += 1;
            if processed % 1_000_000_000 == 0 {
                println!(
                    "{answer}: {}B... ({:.1}%) ({:.3}s)",
                    processed / 1_000_000_000,
                    (root_index as f32 / words.len() as f32) * 100.,
                    start.elapsed().as_secs_f32(),
                );
            }

            partial.next_index += 1;

            if outermost {
                root_index = partial.next_index;
            }

            if !partial.state.valid_guess(guess) {
                continue;
            }

            if min_solution.score < partial.score + partial.state.min_score(guess) * (6 - depth) {
                continue;
            }

            let word_score = partial.state.eval_score(guess);
            if min_solution.score < partial.score + word_score * (6 - depth) {
                continue;
            }

            let new_score = partial.score + word_score;
            if partial.state.guesses.len() == 5 && new_score == min_solution.score {
                continue;
            }

            let mut child_state = partial.state.clone();
            child_state.apply(guess);

            let child_partial = PartialSolutionYellow {
                score: new_score,
                next_index: 0,
                state: child_state,
            };

            if child_partial.state.guesses.len() == 6 {
                let seconds = start.elapsed().as_secs_f32();
                println!(
                    "Found solution (score {}) in {} guesses in {}s ({}M/s)",
                    child_partial.score,
                    processed,
                    seconds as u64,
                    (processed as f32 / 1_000_000. / seconds) as u64
                );
                for word in child_partial.state.guesses {
                    println!("{word}");
                }
                min_solution = SolutionYellow {
                    score: child_partial.score,
                    state: child_partial.state,
                };

                // see same comment in find_min_min_solution
                if new_score <= 1 {
                    break 'outer;
                }

                continue;
            } else {
                stack.push(child_partial);
                continue 'outer;
            }
        }

        stack.pop();
    }

    min_solution
}

fn find_min_solution(answer: Word, words: &[Word], better_than: Solution) -> SolutionSingle {
    let start = Instant::now();

    let mut stack = Vec::new();
    stack.push(PartialSolutionSingle {
        next_index: 0,
        score: 0,
        state: State::new(answer),
    });

    let mut processed = 0usize;
    let mut min_solution = SolutionSingle {
        score: better_than.scores_at[5],
        state: better_than.state,
    };

    let mut root_index = 0;

    'outer: while let Some(partial) = stack.last_mut() {
        let depth = partial.state.guesses.len() as i32;
        let outermost = depth == 0;

        while let Some(&guess) = words.get(partial.next_index as usize) {
            processed += 1;
            if processed % 1_000_000_000 == 0 {
                println!(
                    "{answer}: {}B... ({:.1}%) ({:.3}s)",
                    processed / 1_000_000_000,
                    (root_index as f32 / words.len() as f32) * 100.,
                    start.elapsed().as_secs_f32(),
                );
            }

            partial.next_index += 1;

            if outermost {
                root_index = partial.next_index;
            }

            if !partial.state.valid_guess(guess) {
                continue;
            }

            if min_solution.score < partial.score + partial.state.min_score(guess) * (6 - depth) {
                continue;
            }

            let word_score = partial.state.eval_score(guess);
            if min_solution.score < partial.score + word_score * (6 - depth) {
                continue;
            }

            let new_score = partial.score + word_score;
            if partial.state.guesses.len() == 5 && new_score == min_solution.score {
                continue;
            }

            let mut child_state = partial.state.clone();
            child_state.apply(guess);

            let child_partial = PartialSolutionSingle {
                score: new_score,
                next_index: 0,
                state: child_state,
            };

            if child_partial.state.guesses.len() == 6 {
                let seconds = start.elapsed().as_secs_f32();
                println!(
                    "Found solution (score {}) in {} guesses in {}s ({}M/s)",
                    child_partial.score,
                    processed,
                    seconds as u64,
                    (processed as f32 / 1_000_000. / seconds) as u64
                );
                for word in child_partial.state.guesses {
                    println!("{word}");
                }
                min_solution = SolutionSingle {
                    score: child_partial.score,
                    state: child_partial.state,
                };

                // see same comment in find_min_min_solution
                if new_score <= 1 {
                    break 'outer;
                }

                continue;
            } else {
                stack.push(child_partial);
                continue 'outer;
            }
        }

        stack.pop();
    }

    min_solution
}

#[derive(Copy, Clone)]
struct Solution {
    scores_at: [i32; 6],
    state: State,
}

struct SolutionYellow {
    score: i32,
    state: YellowState,
}

struct SolutionSingle {
    score: i32,
    state: State,
}

struct PartialSolutionZero {
    next_index: u32,
    state: ZeroState,
}

struct PartialSolutionYellow {
    next_index: u32,
    score: i32,
    state: YellowState,
}

struct PartialSolutionSingle {
    next_index: u32,
    score: i32,
    state: State,
}

#[derive(Copy, Clone)]
struct Word {
    mask: u32,
    letters: [u8; 5],
}

impl PartialEq for Word {
    fn eq(&self, other: &Self) -> bool {
        self.letters == other.letters
    }
}

impl Eq for Word {}

impl PartialOrd for Word {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Word {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.mask.count_ones().cmp(&other.mask.count_ones()) {
            Ordering::Equal => {}
            cmp => return cmp,
        }

        self.letters.cmp(&other.letters)
    }
}

impl FromStr for Word {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // assert!(s.len() == 5 && s.is_ascii());
        if s.len() != 5 || !s.is_ascii() {
            return Err(());
        }

        let ascii_letters: [u8; 5] = s.as_bytes()[0..5].try_into().unwrap();
        let letters = ascii_letters.map(|c| c.to_ascii_lowercase() - 'a' as u8);

        let mask = {
            let mut mask = 0;
            for letter in letters {
                mask |= 1 << letter;
            }
            mask
        };

        Ok(Word { mask, letters })
    }
}

impl Display for Word {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for l in self.letters {
            f.write_char((l + 'A' as u8) as char)?;
        }

        Ok(())
    }
}

#[derive(Copy, Clone)]
struct ZeroState {
    gray_mask: u32,
    guesses: GuessVec,
}

impl ZeroState {
    fn new(word: Word) -> Self {
        Self {
            gray_mask: word.mask,
            guesses: Default::default(),
        }
    }

    fn apply(&mut self, guess: Word) {
        self.gray_mask |= guess.mask;
        self.guesses.push(guess);
    }

    fn valid_guess(&self, guess: Word) -> bool {
        (self.gray_mask & guess.mask) == 0
    }
}

#[derive(Copy, Clone)]
struct YellowState {
    word: Word,
    gray_mask: u32,
    yellow_masks: [u32; 5],
    yellow_counts: [i8; 26],
    guesses: GuessVec,
}

impl YellowState {
    fn new(word: Word) -> Self {
        Self {
            word,
            gray_mask: 0,
            yellow_masks: [0; 5],
            yellow_counts: [0; 26],
            guesses: Default::default(),
        }
    }

    fn apply(&mut self, guess: Word) {
        let mut answer_letters = self.word.letters;
        let mut guess_letters = guess.letters;

        let mut yellow_counts = [0i8; 26];

        for i in 0..5 {
            let guess_letter = guess_letters[i];

            if let Some(o) = answer_letters.into_iter().position(|l| l == guess_letter) {
                self.yellow_masks[i] |= 1 << guess_letter;
                answer_letters[o] = !0;
                guess_letters[i] = !0;
                yellow_counts[guess_letter as usize] += 1;
            } else if self.word.letters.contains(&guess_letter) {
                self.yellow_masks[i] |= 1 << guess_letter;
            }
        }

        for l in guess_letters {
            if l != !0 {
                self.gray_mask |= 1 << l;
            }
        }

        #[cfg(debug_assertions)]
        for (current, new) in self.yellow_counts.into_iter().zip(yellow_counts) {
            if new < current {
                panic!("Not all yellows used");
            }
        }
        self.yellow_counts = yellow_counts;

        self.guesses.push(guess);
    }

    fn min_score(&self, guess: Word) -> i32 {
        (self.word.mask & guess.mask).count_ones() as _
    }
    fn eval_score(&self, guess: Word) -> i32 {
        let mut answer_letters = self.word.letters;

        let mut score = 0;

        for i in 0..5 {
            let guess_letter = guess.letters[i];

            if let Some(o) = answer_letters.into_iter().position(|l| l == guess_letter) {
                score += 1;
                answer_letters[o] = !0;
            }
        }

        score
    }

    fn valid_guess(&self, guess: Word) -> bool {
        let mut answer_letters = self.word.letters;
        let mut yellow_counts = self.yellow_counts;

        for i in 0..5 {
            let guess_letter = guess.letters[i];

            if (self.yellow_masks[i] & (1 << guess_letter)) != 0 {
                return false;
            }

            if let Some(o) = answer_letters.into_iter().position(|l| l == guess_letter) {
                answer_letters[o] = !0;
                yellow_counts[guess_letter as usize] -= 1;
            } else if (self.gray_mask & (1 << guess_letter)) != 0 {
                return false;
            }
        }

        if yellow_counts.into_iter().any(|c| c > 0) {
            return false;
        }

        !self.guesses.contains(&guess)
    }
}

#[derive(Copy, Clone)]
struct State {
    word: Word,
    gray_mask: u32,
    green_index_mask: u8,
    yellow_masks: [u32; 5],
    yellow_counts: [i8; 26],
    guesses: GuessVec,
}

impl State {
    fn new(word: Word) -> Self {
        Self {
            word,
            gray_mask: 0,
            green_index_mask: 0,
            yellow_masks: [0; 5],
            yellow_counts: [0; 26],
            guesses: Default::default(),
        }
    }

    fn apply(&mut self, guess: Word) {
        let mut answer_letters = self.word.letters;
        let mut guess_letters = guess.letters;

        let mut yellow_counts = [0i8; 26];

        for i in 0..5 {
            let guess_letter = guess_letters[i];

            if guess_letter == answer_letters[i] {
                self.green_index_mask |= 1 << i;
                yellow_counts[guess_letter as usize] += 1;
                answer_letters[i] = !0;
                guess_letters[i] = !0;
            }
        }

        for i in 0..5 {
            let guess_letter = guess_letters[i];

            if guess_letter == !0 {
                continue;
            }

            if let Some(o) = answer_letters.into_iter().position(|l| l == guess_letter) {
                self.yellow_masks[i] |= 1 << guess_letter;
                answer_letters[o] = !0;
                guess_letters[i] = !0;
                yellow_counts[guess_letter as usize] += 1;
            } else if self.word.letters.contains(&guess_letter) {
                self.yellow_masks[i] |= 1 << guess_letter;
            }
        }

        for l in guess_letters {
            if l != !0 {
                self.gray_mask |= 1 << l;
            }
        }

        #[cfg(debug_assertions)]
        for (current, new) in self.yellow_counts.into_iter().zip(yellow_counts) {
            if new < current {
                panic!("Not all yellows used");
            }
        }
        self.yellow_counts = yellow_counts;

        self.guesses.push(guess);
    }

    fn min_score(&self, guess: Word) -> i32 {
        (self.word.mask & guess.mask).count_ones() as _
    }

    fn eval_score(&self, guess: Word) -> i32 {
        let mut answer_letters = self.word.letters;
        let mut guess_letters = guess.letters;

        let mut score = 0;

        for i in 0..5 {
            let guess_letter = guess_letters[i];

            if guess_letter == answer_letters[i] {
                score += 100;
                answer_letters[i] = !0;
                guess_letters[i] = !0;
            }
        }

        for i in 0..5 {
            let guess_letter = guess_letters[i];

            if guess_letter == !0 {
                continue;
            }

            if let Some(o) = answer_letters.into_iter().position(|l| l == guess_letter) {
                score += 1;
                answer_letters[o] = !0;
            }
        }

        score
    }

    fn valid_guess(&self, guess: Word) -> bool {
        let mut answer_letters = self.word.letters;
        let mut guess_letters = guess.letters;

        let mut yellow_counts = self.yellow_counts;
        let mut remaining_greens = self.green_index_mask;
        for i in 0..5 {
            let guess_letter = guess.letters[i];

            if guess_letter == answer_letters[i] {
                answer_letters[i] = !0;
                guess_letters[i] = !0;
                remaining_greens &= !(1 << i);
                yellow_counts[guess_letter as usize] -= 1;
            }
        }

        if remaining_greens != 0 {
            return false;
        }

        for i in 0..5 {
            let guess_letter = guess_letters[i];

            if guess_letter == !0 {
                continue;
            }

            if (self.yellow_masks[i] & (1 << guess_letter)) != 0 {
                return false;
            }

            if let Some(o) = answer_letters.into_iter().position(|l| l == guess_letter) {
                answer_letters[o] = !0;
                yellow_counts[guess_letter as usize] -= 1;
            } else if (self.gray_mask & (1 << guess_letter)) != 0 {
                return false;
            }
        }

        if yellow_counts.into_iter().any(|c| c > 0) {
            return false;
        }

        !self.guesses.contains(&guess)
    }
}

#[derive(Copy, Clone)]
struct GuessVec {
    len: u8,
    guesses: [Word; 6],
}

impl Default for GuessVec {
    fn default() -> Self {
        Self {
            len: 0,
            guesses: [Word {
                mask: 0,
                letters: [0; 5],
            }; 6],
        }
    }
}

impl PartialEq for GuessVec {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.as_slice() == other.as_slice()
    }
}

impl Eq for GuessVec {}

impl PartialOrd for GuessVec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GuessVec {
    fn cmp(&self, other: &Self) -> Ordering {
        for (a, b) in self.into_iter().zip(*other) {
            match a.cmp(&b) {
                Ordering::Equal => {}
                cmp => return cmp,
            }
        }

        self.len.cmp(&other.len)
    }
}

impl Index<usize> for GuessVec {
    type Output = Word;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl GuessVec {
    fn len(&self) -> usize {
        self.len as _
    }

    fn push(&mut self, word: Word) {
        assert!(self.len < 6, "A GuessVec can only store up to 6 elements");
        self.guesses[self.len as usize] = word;
        self.len += 1;
    }

    fn contains(&self, word: &Word) -> bool {
        self.guesses[..self.len as usize].contains(word)
    }

    fn as_slice(&self) -> &[Word] {
        &self.guesses[..self.len as usize]
    }
}

impl IntoIterator for GuessVec {
    type Item = Word;
    type IntoIter = GuessVecIter;

    fn into_iter(self) -> Self::IntoIter {
        GuessVecIter {
            vec: self,
            next_idx: 0,
        }
    }
}

struct GuessVecIter {
    vec: GuessVec,
    next_idx: u8,
}

impl Iterator for GuessVecIter {
    type Item = Word;

    fn next(&mut self) -> Option<Self::Item> {
        (self.next_idx < self.vec.len).then(|| {
            let idx = self.next_idx as usize;
            self.next_idx += 1;
            self.vec.guesses[idx]
        })
    }
}

impl FusedIterator for GuessVecIter {}

#[cfg(test)]
mod test {
    use super::*;

    fn word(word: &str) -> Word {
        Word::from_str(word).unwrap()
    }

    #[test]
    fn valid_guesses() {
        let mut partial = State::new(word("abcde"));
        assert!(partial.valid_guess(word("fghij")));
        partial.apply(word("fghij"));
        assert!(partial.valid_guess(word("bcdea")));
        partial.apply(word("bcdea"));
        assert!(partial.valid_guess(word("cdeab")));
        partial.apply(word("cdeab"));
        assert!(partial.valid_guess(word("deabc")));
        partial.apply(word("deabc"));
        assert!(partial.valid_guess(word("eabcd")));
        partial.apply(word("eabcd"));
        assert!(partial.valid_guess(word("abcde")));
        partial.apply(word("abcde"));
    }

    #[test]
    fn must_use_green() {
        let mut partial = State::new(word("abcde"));
        partial.apply(word("aghij"));
        assert!(!partial.valid_guess(word("klmno")));
        assert!(partial.valid_guess(word("almno")));
    }

    #[test]
    fn must_use_yellow() {
        let mut partial = State::new(word("abcde"));
        partial.apply(word("fahij"));
        assert!(!partial.valid_guess(word("klmno")));
        assert!(!partial.valid_guess(word("kamno")));
        assert!(partial.valid_guess(word("klano")));
    }

    #[test]
    fn cant_use_grey() {
        let mut partial = State::new(word("abcde"));
        partial.apply(word("fghij"));
        assert!(partial.valid_guess(word("klmno")));
        assert!(!partial.valid_guess(word("kfmno")));
    }

    #[test]
    fn double_letters() {
        let partial = State::new(word("abbcd"));
        assert_eq!(partial.eval_score(word("bfgbb")), 2);
        assert_eq!(partial.eval_score(word("efbhi")), 100);
        assert_eq!(partial.eval_score(word("bfbhi")), 101);
        assert_eq!(partial.eval_score(word("bbbhi")), 200);
    }

    #[test]
    fn double_yellow() {
        let mut partial = State::new(word("aacde"));
        partial.apply(word("fgaaj"));
        assert!(!partial.valid_guess(word("klmna")));
        assert!(!partial.valid_guess(word("klmaa")));
        assert!(partial.valid_guess(word("almna")));
        assert!(partial.valid_guess(word("aamna")));
    }

    #[test]
    fn yellow_double() {
        let mut partial = State::new(word("abcde"));
        partial.apply(word("faaij"));
        assert!(!partial.valid_guess(word("kamno")));
        assert!(!partial.valid_guess(word("klano")));
        assert!(partial.valid_guess(word("klmao")));
    }

    #[test]
    fn grey_yellow() {
        let mut partial = State::new(word("abcde"));
        assert_eq!(partial.eval_score(word("fgbbj")), 1);
        partial.apply(word("fgbbj"));
        assert!(!partial.valid_guess(word("klmbo")));
    }

    #[test]
    fn cant_repeat_guesses() {
        let mut partial = State::new(word("abcde"));
        partial.apply(word("fghij"));
        assert!(!partial.valid_guess(word("fghij")));
    }

    #[test]
    fn patio_solution_invalid() {
        let mut state = State::new(word("patio"));
        state.apply(word("esses"));
        state.apply(word("allyl"));
        state.apply(word("bubba"));
        state.apply(word("chack"));
        state.apply(word("ngram"));
        assert!(!state.valid_guess(word("jaffa")));
    }
}
