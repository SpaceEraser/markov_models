use crate::bitmanip::BitMatrix;
use crate::pick_from_weighted_uniform;
use image::GrayImage;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct ImageModel {
    prefix_size: usize,
    distribution: HashMap<BitMatrix, HashMap<bool, f64>>,
}

impl ImageModel {
    pub fn from_images<I: Iterator<Item = GrayImage>>(iter: I, prefix_size: usize) -> Self {
        Self::from_bitmatrixes(iter.map(|im| im.into()), prefix_size)
    }

    pub fn from_bitmatrixes<I: Iterator<Item = BitMatrix>>(iter: I, prefix_size: usize) -> Self {
        assert!(prefix_size > 1);

        let mut this = Self {
            prefix_size,
            distribution: Default::default(),
        };
        for im in iter {
            this.raw_analyze_image(&im.into());
        }
        this.normalize();
        return this;
    }

    pub fn generate_at_most(&self, width: usize, height: usize) -> BitMatrix {
        let mut im = BitMatrix::new(width, height);
        let mut prefix_buf = BitMatrix::from_val(self.prefix_size, self.prefix_size, true);

        for x in 0..width {
            for y in 0..height {
                self.fill_prefix_buf(x, y, &im, &mut prefix_buf);

                if let Some(distr) = self.distribution.get(&prefix_buf) {
                    let i = pick_from_weighted_uniform(distr.values().cloned());
                    let suffix = *distr.keys().nth(i).unwrap();
                    im.set([x, y], suffix);
                }
            }
        }

        return im;
    }

    fn fill_prefix_buf(&self, x: usize, y: usize, im: &BitMatrix, prefix_buf: &mut BitMatrix) {
        for px in 0..self.prefix_size {
            for py in 0..self.prefix_size {
                if (px == self.prefix_size - 1 && py == self.prefix_size - 1)
                    || x + px < self.prefix_size
                    || x + px - self.prefix_size >= im.width()
                    || y + py < self.prefix_size
                    || y + py - self.prefix_size >= im.height()
                {
                    prefix_buf.set([px, py], true);
                } else {
                    prefix_buf.set(
                        [px, py],
                        im.get([x + px - self.prefix_size, y + py - self.prefix_size]),
                    )
                }
            }
        }
    }

    fn raw_analyze_image(&mut self, im: &BitMatrix) {
        for x in 0..im.width() + self.prefix_size {
            for y in 0..im.height() + self.prefix_size {
                let mut prefix_buf = BitMatrix::new(self.prefix_size, self.prefix_size);
                self.fill_prefix_buf(x, y, im, &mut prefix_buf);

                let suffix = if x >= im.width() || y >= im.height() {
                    true
                } else {
                    im.get([x, y])
                };

                *self
                    .distribution
                    .entry(prefix_buf)
                    .or_default()
                    .entry(suffix)
                    .or_default() += 1.0;
            }
        }
    }

    fn normalize(&mut self) {
        for (_, v) in self.distribution.iter_mut() {
            let sum: f64 = v.values().sum();
            for (_, i) in v {
                *i /= sum;
            }
        }
    }
}
