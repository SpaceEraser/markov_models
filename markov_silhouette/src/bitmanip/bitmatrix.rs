use image::{GrayImage, Luma};

use super::bitvec::BitVec;
use serde::{Serialize, Deserialize};

#[derive(Clone, Hash, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct BitMatrix {
    width: usize,
    height: usize,
    vec: BitVec,
}

impl BitMatrix {
    pub fn new(width: usize, height: usize) -> Self {
        Self::from_val(width, height, false)
    }

    pub fn from_val(width: usize, height: usize, val: bool) -> Self {
        Self {
            width,
            height,
            vec: BitVec::new_val(width * height, val),
        }
    }

    pub fn dimensions(&self) -> [usize; 2] {
        [self.width, self.height]
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn get(&self, index: [usize; 2]) -> bool {
        self.vec.get(self.to_1d(index)).unwrap_or_else(|| {
            if index[0] >= self.width {
                panic!("index[0] out of bounds: {} >= {}", index[0], self.width)
            } else {
                panic!("index[1] out of bounds: {} >= {}", index[1], self.height)
            }
        })
    }

    pub fn set(&mut self, index: [usize; 2], v: bool) {
        let i = self.to_1d(index);
        self.vec.set(i, v).unwrap_or_else(|| {
            if index[0] >= self.width {
                panic!("index[0] out of bounds: {} >= {}", index[0], self.width)
            } else {
                panic!("index[1] out of bounds: {} >= {}", index[1], self.height)
            }
        });
    }

    fn to_1d(&self, index: [usize; 2]) -> usize {
        index[0] + self.width * index[1]
    }

    pub fn is_true(&self) -> bool {
        if self.width == 0 || self.height == 0 {
            return true;
        }
        for i in 0..self.vec.len() {
            if !self.vec.get(i).unwrap() {
                return false;
            }
        }
        return true;
    }
}

impl From<&[&[u8]]> for BitMatrix {
    fn from(mat: &[&[u8]]) -> Self {
        let height = mat.len();
        let width = if height == 0 { 0 } else { mat[0].len() };
        let mut this = BitMatrix::new(width, height);
        for y in 0..height {
            assert_eq!(mat[y].len(), width);
            for x in 0..width {
                this.set([x, y], mat[y][x] > 0);
            }
        }
        return this;
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> From<&[&[u8; WIDTH]; HEIGHT]> for BitMatrix {
    fn from(mat: &[&[u8; WIDTH]; HEIGHT]) -> Self {
        let mut this = BitMatrix::new(WIDTH, HEIGHT);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                this.set([x, y], mat[y][x] > 0);
            }
        }
        return this;
    }
}

impl From<GrayImage> for BitMatrix {
    fn from(image: GrayImage) -> Self {
        BitMatrix::from(&image)
    }
}

impl From<&GrayImage> for BitMatrix {
    fn from(image: &GrayImage) -> Self {
        let mut this = BitMatrix::new(image.width() as _, image.height() as _);
        for x in 0..this.width {
            for y in 0..this.height {
                this.set([x, y], image[(x as _, y as _)].0[0] > 127)
            }
        }
        return this;
    }
}

impl Into<GrayImage> for &BitMatrix {
    fn into(self) -> GrayImage {
        GrayImage::from_fn(self.width as _, self.height as _, |x, y| {
            if self.get([x as _, y as _]) {
                Luma([255])
            } else {
                Luma([0])
            }
        })
    }
}

impl Into<GrayImage> for BitMatrix {
    fn into(self) -> GrayImage {
        (&self).into()
    }
}

impl std::fmt::Debug for BitMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BitMatrix {{ {}×{}, [", self.width, self.height)?;
        for y in 0..self.height {
            for x in 0..self.width {
                write!(f, "{}", if self.get([x, y]) { "1" } else { "0" })?;
            }
            if y != self.height - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "] }}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_matrix() {
        let mut bitmat = BitMatrix::new(2, 3);
        assert_eq!(bitmat.vec.len(), 2 * 3);
        assert_eq!(bitmat.width(), 2);
        assert_eq!(bitmat.height(), 3);

        assert_eq!(bitmat.to_1d([1, 2]), 1 + 2 * 2);

        bitmat.set([1, 2], true);
        assert_eq!(bitmat.get([1, 2]), true);
    }
}
