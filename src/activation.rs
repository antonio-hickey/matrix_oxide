use crate::Matrix;
use std::f64::consts::PI;
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

    /// Apply the GeLU activation function onto a `Matrix`
    /// NOTE: Smoother (near 0) than ReLU & potential for regularization effects.
    pub fn gelu(&self) -> Matrix<T>
    where
        T: Copy + PartialOrd + Default + From<f64> + Into<f64>,
    {
        let data: Vec<T> = self
            .data
            .iter()
            .map(|&x| {
                let x_f64: f64 = x.into();
                let x_gelu = 0.5
                    * x_f64
                    * (1.0 + ((2.0 / PI).sqrt() * (x_f64 + 0.04715 * x_f64.powi(3))).tanh());
                T::from(x_gelu)
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

    #[test]
    /// Verify each element is correctly GeLU'd
    fn test_gelu() {
        let matrix = Matrix {
            data: vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0],
            row_size: 3,
            col_size: 3,
        };

        let expected = [
            0.841344746,
            -0.045500263,
            2.9963625,
            -0.0003058,
            4.9998675,
            -0.0000009,
            6.999999998,
            -0.000000001,
            8.9999999,
        ];
        let result = matrix.gelu();

        for (res, exp) in result.data.iter().zip(expected.iter()) {
            // NOTE: due to inaccuracy in floating point arithmetic this has a tolerance of 0.01
            assert!((Into::<f64>::into(*res) - exp).abs() < 1e-2);
        }
    }
}
