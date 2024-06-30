use crate::Matrix;
use std::ops::{Add, Mul};

impl<T> Matrix<T>
where
    T: PartialOrd + Default + Copy + Mul<Output = T>,
{
    /// Apply the ReLU activation function onto a `Matrix`
    pub fn relu(&self) -> Matrix<T>
    where
        T: PartialOrd + Default + Copy,
    {
        let data: Vec<T> = self
            .data
            .iter()
            .map(|&x| if x < T::default() { T::default() } else { x })
            .collect();

        Matrix {
            data,
            row_size: self.row_size,
            col_size: self.col_size,
        }
    }

    /// Apply the Leaky ReLU activation function onto a `Matrix`
    pub fn leaky_relu(&self, alpha: T) -> Matrix<T>
    where
        T: PartialOrd + Default + Copy + Mul<Output = T>,
    {
        let data: Vec<T> = self
            .data
            .iter()
            .map(|&x| if x < T::default() { x * alpha } else { x })
            .collect();

        Matrix {
            data,
            row_size: self.row_size,
            col_size: self.col_size,
        }
    }

    /// Apply backward pass for the ReLU activation function onto a `Matrix`
    pub fn relu_backward(&self) -> Matrix<T>
    where
        T: Copy + PartialOrd + Default + Add<T, Output = T> + Mul<T, Output = T> + From<u8>,
    {
        let data: Vec<T> = self
            .data
            .iter()
            .map(|&x| {
                if x >= T::default() {
                    T::default() + T::from(1u8)
                } else {
                    T::default()
                }
            })
            .collect();

        Matrix {
            data,
            row_size: self.row_size,
            col_size: self.col_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Verify each element is correctly ReLU'd
    fn test_relu() {
        let matrix = Matrix {
            data: vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0],
            row_size: 3,
            col_size: 3,
        };

        let expected = vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0];
        let result = matrix.relu();

        assert_eq!(result.data, expected);
    }

    #[test]
    /// Verify each element is correctly Leaky ReLU'd
    fn test_leaky_relu() {
        let matrix = Matrix {
            data: vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0],
            row_size: 3,
            col_size: 3,
        };

        let alpha = 0.1;

        let expected = vec![
            1.0,
            -0.2,
            3.0,
            -0.4,
            5.0,
            -0.6000000000000001,
            7.0,
            -0.8,
            9.0,
        ];
        let result = matrix.leaky_relu(alpha);

        assert_eq!(result.data, expected);
    }

    #[test]
    /// Verify each element in the gradient matrix
    fn test_relu_backward() {
        let matrix = Matrix {
            data: vec![1, -2, 3, -4, 5, -6, 7, -8, 9],
            row_size: 3,
            col_size: 3,
        };

        let expected = vec![1, 0, 1, 0, 1, 0, 1, 0, 1];
        let result = matrix.relu_backward();

        assert_eq!(result.data, expected);
    }
}
