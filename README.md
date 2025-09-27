# Wrong Wordle

A program to solve wordle "the wrong way", i.e. getting as few yellow/greens as possible (under hard mode rules).

Inspired by https://www.youtube.com/watch?v=zoh5eLOjwHA

## Usage

`cargo run --release -- <answer>` to solve "\<answer>" using the word list in `words.txt`.

`cargo run --release` to solve all answers in `answers.txt` using the word list in `words.txt` (using all CPU cores).

`cargo run --release -- -j <n>` to solve all answers in `answers.txt` using the word list in `words.txt` using `n` CPU cores.
