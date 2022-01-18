#![allow(incomplete_features)]
#![feature(const_generics)]

mod ngram_model;

use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand::rngs::SmallRng::from_entropy();
    // let text = std::fs::read_to_string("shakespeare.txt")
    let text = std::fs::read_to_string("big.txt").expect("Can't find text file");
    let model = ngram_model::NGramModel::<3>::from_text(&*text);
    // let model = NGramModel::<2>::from_text("abc");
    // println!("{:?}", model);
    for _ in 0..50 {
        dbg!(model.generate_exact(rng.gen_range(3..=10)));
    }
}
