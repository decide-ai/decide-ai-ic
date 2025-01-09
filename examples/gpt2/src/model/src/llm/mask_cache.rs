use std::collections::HashSet;
use candle::{DType, Device, Tensor};
use crate::llm::gpt2::{MaskCache};
//use anyhow::{Result, Error as E};
use anyhow::{Result};

pub struct VecMaskCache {
    masks: Vec<Tensor>,
    device: Device,
    num_heads: usize,
    cached_sizes: HashSet<usize>,
}

impl MaskCache for VecMaskCache {
    fn get_mask(&self, size: usize, batch_size: usize) -> candle::Result<Tensor> {
        if size < self.masks.len() {
            if batch_size == 1 {
                Ok(self.masks[size].clone())
            } else {
                self.masks[size].expand(&[batch_size, self.num_heads, size, size])
            }
        } else {
            let mask = Self::create_causal_mask(size, self.num_heads, &self.device)?;
            mask.expand(&[batch_size, self.num_heads, size, size])
        }
    }

    fn cached_sizes(&self) -> &HashSet<usize> {
        &self.cached_sizes
    }


}

impl VecMaskCache {
    pub fn new(max_size: usize, num_heads: usize, device: Device) -> Result<Self> {
        let mut masks = Vec::with_capacity(max_size + 1);
        masks.push(Tensor::zeros((1, 1, 1, 1), DType::U8, &device)?);
        for size in 1..=max_size {
            let mask = Self::create_causal_mask(size, num_heads, &device)?;
            masks.push(mask);
        }
        let cached_sizes = (1..=max_size).collect::<HashSet<usize>>();

        Ok(Self { masks, device, num_heads, cached_sizes })
    }

    fn create_causal_mask(size: usize, num_heads: usize, device: &Device) -> candle::Result<Tensor> {
        let mask: Vec<_> = (0..size)
            .flat_map(|i| (0..size).map(move |j| u8::from(j <= i)))
            .collect();
        Tensor::from_slice(&mask, (size, size), device)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .expand((1, num_heads, size, size))?
            .to_dtype(DType::U8)
    }
}