use std::fmt::{self, Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::str::FromStr;

pub trait MatrixData<T>:
    Mul<Output = T>
    + Display
    + Add<Output = T>
    + Div<Output = T>
    + Neg<Output = T>
    + Copy
    + Debug
    + PartialEq
    + PartialOrd
    + AddAssign
    + Sub<Output = T>
    + SubAssign
    + Sum
    + std::marker::Send
    + std::marker::Sync
{
    fn zero() -> T;
    fn one() -> T;
    fn ln(self) -> T;
    fn exp(self) -> T;
}
impl MatrixData<f64> for f64 {
    fn zero() -> f64 {
        0.0
    }
    fn one() -> f64 {
        1.0
    }
    fn ln(self) -> f64 {
        self.ln()
    }
    fn exp(self) -> f64 {
        self.exp()
    }
}

impl MatrixData<f32> for f32 {
    fn zero() -> f32 {
        0.0
    }
    fn one() -> f32 {
        1.0
    }
    fn ln(self) -> f32 {
        self.ln()
    }
    fn exp(self) -> f32 {
        self.exp()
    }
}
impl MatrixData<i64> for i64 {
    fn zero() -> i64 {
        0
    }
    fn one() -> i64 {
        1
    }
    fn ln(self) -> i64 {
        (self as f64).ln() as i64
    }
    fn exp(self) -> i64 {
        (self as f64).exp() as i64
    }
}
impl MatrixData<i32> for i32 {
    fn zero() -> i32 {
        0
    }
    fn one() -> i32 {
        1
    }
    fn ln(self) -> i32 {
        (self as f64).ln() as i32
    }
    fn exp(self) -> i32 {
        (self as f64).exp() as i32
    }
}

// impl<T> MatrixData<T> for f64 where
//     T: Mul<Output = T>
//         + Add<Output = T>
//         + Div<Output = T>
//         + Neg<Output = T>
//         + Copy
//         + Debug
//         + PartialEq
//         + PartialOrd
//         + AddAssign
//         + Sub<Output = T>
//         + SubAssign
// {

// }

// Simple Contigious Matrix
// This will likely be too generic for our needs
pub struct Matrix<'a, T> {
    data: &'a [T],
    pub index: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    stride1: usize,
    stride2: usize,
}

impl<'a, T> Matrix<'a, T>
where
    T: MatrixData<T>,
{
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        Matrix {
            data,
            index: (0..rows).collect(),
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
        idx += j * self.stride1;
        idx
    }

    pub fn get_col_slice(&self, col: usize, start_row: usize, end_row: usize) -> &[T] {
        let i = self.item_index(start_row, col);
        let j = self.item_index(end_row, col);
        &self.data[i..j]
    }

    pub fn get_col(&self, col: usize) -> &[T] {
        self.get_col_slice(col, 0, self.rows)
    }
}

impl<'a, T> fmt::Display for Matrix<'a, T>
where
    T: FromStr + std::fmt::Display + MatrixData<T>,
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
    #[test]
    fn test_get_col_slice() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 3, 2);
        assert_eq!(m.get_col_slice(0, 0, 3), &vec![1, 2, 3]);
        assert_eq!(m.get_col_slice(1, 0, 2), &vec![5, 6]);
        assert_eq!(m.get_col_slice(1, 1, 3), &vec![6, 7]);
        assert_eq!(m.get_col_slice(0, 1, 2), &vec![2]);
    }

    #[test]
    fn test_get_col() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 3, 2);
        assert_eq!(m.get_col(1), &vec![5, 6, 7]);
    }
}
