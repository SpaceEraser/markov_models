use rand::{Rng, SeedableRng};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct NGramModel<const N: usize> {
    char_distr: HashMap<[u8; N], HashMap<u8, f64>>,
}

impl<const N: usize> NGramModel<N> {
    pub fn from_text(text: &str) -> Self {
        let mut this = Self::default();
        this.analyze(text);
        return this;
    }

    pub fn generate_exact(&self, len: usize) -> String {
        loop {
            let s = self.generate_at_most(len);
            if s.len() == len {
                return s;
            }
        }
    }

    pub fn generate_at_most(&self, len: usize) -> String {
        let mut ret = String::with_capacity(len);
        let mut rng = rand::rngs::SmallRng::from_entropy();
        let mut buf = [255u8; N];

        for _ in 0..len {
            if let Some(distr) = self.char_distr.get(&buf) {
                let f: f64 = rng.gen();
                let mut s = 0.0;
                let mut c = 255;
                for (&k, &v) in distr {
                    s += v;
                    if f <= s {
                        c = k;
                        break;
                    }
                }
                ret.push(c as char);

                for i in 0..buf.len() - 1 {
                    buf[i] = buf[i + 1];
                }
                buf[buf.len() - 1] = c as u8;
            }

            if buf.iter().all(|&c| c == 255) {
                break;
            }
        }

        return ret.trim_matches(255 as char).to_string();
    }

    fn analyze(&mut self, string: &str) {
        let re = regex::Regex::new(r"[a-zA-Z-']+").expect("Failed to build regex");
        for m in re.find_iter(string) {
            self.analyze_word(m.as_str());
        }

        for (_, v) in self.char_distr.iter_mut() {
            let sum: f64 = v.values().sum();
            for (_, i) in v {
                *i /= sum;
            }
        }
    }

    fn analyze_word(&mut self, word: &str) {
        let word = word.as_bytes();
        let mut buf = [0u8; N];

        for i in 0..word.len() + N {
            let l = if i < word.len() {
                word[i].to_ascii_lowercase()
            } else {
                255
            };

            for j in 0..N {
                buf[j] = if i + j < N || i + j - N >= word.len() {
                    255
                } else {
                    word[i + j - N].to_ascii_lowercase()
                };
            }

            self.add_ngram(buf, l);
        }
    }

    fn add_ngram(&mut self, k: [u8; N], l: u8) {
        *self.char_distr.entry(k).or_default().entry(l).or_default() += 1.0;
    }
}
