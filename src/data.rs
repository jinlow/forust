use std::fmt::{self, Debug};
use std::str::FromStr;

// Simple Contigious Matrix
// This will likely be too generic for out needs
pub struct Matrix<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    stride1: usize,
    stride2: usize,
}

impl<'a, T> Matrix<'a, T> 
where
    T: std::fmt::Debug
{
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        Matrix {
            data,
            rows,
            cols,
            stride1: rows,
            stride2: 1,
        }
    }

    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[self.item_index(i, j)]
    }

    fn item_index(&self, i: usize, j: usize) -> usize {
        let mut idx: usize;
        idx = self.stride2 * i;
        idx = idx + (j * self.stride1);
        idx
    }

    pub fn get_col_slice(&self, col: usize, start_row: usize, end_row: usize) -> &[T]{
        let i = self.item_index(start_row, col);
        let j = self.item_index(end_row, col);
        println!("{:?}", self.data[i]);
        println!("{:?}", self.data[j]);
        
        &self.data[i..j]
    }
}

impl<'a, T> fmt::Display for Matrix<'a, T>
where
    T: FromStr + std::fmt::Display + Copy + Debug,
    <T as FromStr>::Err: 'static + std::error::Error,
{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut val = String::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                val.push_str(self.get(i, j).to_string().as_str());
                if j == (self.cols - 1) {
                    val.push('\n');
                } else {
                    val.push(' ');
                }
            }
        }
        write!(f, "{}", val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 2, 3);
        println!("{}", m);
        assert_eq!(m.get(0, 0), &1);
        assert_eq!(m.get(1, 0), &2);
    }
    #[test  ]
    fn test_get_col_slice() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 3, 2);
        assert_eq!(m.get_col_slice(0, 0, 3), &vec![1, 2, 3]);
    }
}

