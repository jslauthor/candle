use crate::{DType, Device, Error, Result, Shape, StridedIndex};

// TODO: Think about whether we would be better off with a dtype and
// a buffer as an owned slice of bytes.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda { gpu_id: usize }, // TODO: Actually add the storage.
}

trait UnaryOp {
    const NAME: &'static str;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
}

trait BinaryOp {
    const NAME: &'static str;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
}

struct Add;
struct Div;
struct Mul;
struct Sub;
struct Neg;
struct Sqr;
struct Sqrt;

impl BinaryOp for Add {
    const NAME: &'static str = "add";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 + v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 + v2
    }
}

impl BinaryOp for Sub {
    const NAME: &'static str = "sub";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 - v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 - v2
    }
}

impl BinaryOp for Mul {
    const NAME: &'static str = "mul";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 * v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 * v2
    }
}

impl BinaryOp for Div {
    const NAME: &'static str = "div";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 / v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 / v2
    }
}

impl UnaryOp for Neg {
    const NAME: &'static str = "neg";
    fn f32(v1: f32) -> f32 {
        -v1
    }
    fn f64(v1: f64) -> f64 {
        -v1
    }
}

impl UnaryOp for Sqr {
    const NAME: &'static str = "sqr";
    fn f32(v1: f32) -> f32 {
        v1 * v1
    }
    fn f64(v1: f64) -> f64 {
        v1 * v1
    }
}

impl UnaryOp for Sqrt {
    const NAME: &'static str = "sqrt";
    fn f32(v1: f32) -> f32 {
        v1.sqrt()
    }
    fn f64(v1: f64) -> f64 {
        v1.sqrt()
    }
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Cuda { gpu_id } => Device::Cuda { gpu_id: *gpu_id },
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
            Self::Cuda { .. } => todo!(),
        }
    }

    pub(crate) fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.device();
        let rhs = rhs.device();
        if lhs != rhs {
            Err(Error::DeviceMismatchBinaryOp { lhs, rhs, op })
        } else {
            Ok(())
        }
    }

    pub(crate) fn same_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();
        if lhs != rhs {
            Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op })
        } else {
            Ok(())
        }
    }

    pub(crate) fn affine_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> Result<Self> {
        // TODO: Different code path for the contiguous case?
        match self {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F32(storage) => {
                    let index = StridedIndex::new(shape.dims(), stride);
                    let mul = mul as f32;
                    let add = add as f32;
                    let data = index.map(|i| storage[i] * mul + add).collect();
                    Ok(Storage::Cpu(CpuStorage::F32(data)))
                }
                CpuStorage::F64(storage) => {
                    let index = StridedIndex::new(shape.dims(), stride);
                    let data = index.map(|i| storage[i] * mul + add).collect();
                    Ok(Storage::Cpu(CpuStorage::F64(data)))
                }
            },
            Self::Cuda { .. } => todo!(),
        }
    }

    fn unary_impl<B: UnaryOp>(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        // TODO: Different code path for the contiguous case?
        match self {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F32(storage) => {
                    let index = StridedIndex::new(shape.dims(), stride);
                    let data = index.map(|i| B::f32(storage[i])).collect();
                    Ok(Storage::Cpu(CpuStorage::F32(data)))
                }
                CpuStorage::F64(storage) => {
                    let index = StridedIndex::new(shape.dims(), stride);
                    let data = index.map(|i| B::f64(storage[i])).collect();
                    Ok(Storage::Cpu(CpuStorage::F64(data)))
                }
            },
            Self::Cuda { .. } => todo!(),
        }
    }

    // TODO: Support broadcasting?
    fn binary_impl<B: BinaryOp>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.same_device(rhs, B::NAME)?;
        self.same_dtype(rhs, B::NAME)?;
        // The ggml implementation has different paths based on whether the rhs is contiguous
        // or not, for now we only consider the general case but we should benchmark and do the
        // same if it helps.
        // https://github.com/ggerganov/llama.cpp/blob/aacdbd40562684665b6f7b8ba6695b7a2088bbb0/ggml.c#L7895
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => match (lhs, rhs) {
                (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) => {
                    let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                    let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                    let data = lhs_index
                        .zip(rhs_index)
                        .map(|(lhs_i, rhs_i)| B::f32(lhs[lhs_i], rhs[rhs_i]))
                        .collect();
                    Ok(Storage::Cpu(CpuStorage::F32(data)))
                }
                (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                    let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                    let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                    let data = lhs_index
                        .zip(rhs_index)
                        .map(|(lhs_i, rhs_i)| B::f64(lhs[lhs_i], rhs[rhs_i]))
                        .collect();
                    Ok(Storage::Cpu(CpuStorage::F64(data)))
                }
                _ => {
                    // This should be covered by the dtype check above.
                    Err(Error::DTypeMismatchBinaryOp {
                        lhs: lhs.dtype(),
                        rhs: rhs.dtype(),
                        op: B::NAME,
                    })
                }
            },
            (Self::Cuda { .. }, Self::Cuda { .. }) => todo!(),
            (lhs, rhs) => {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: lhs.device(),
                    rhs: rhs.device(),
                    op: B::NAME,
                })
            }
        }
    }

    pub(crate) fn add_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.binary_impl::<Add>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn sub_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.binary_impl::<Sub>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn mul_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.binary_impl::<Mul>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn div_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.binary_impl::<Div>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn neg_impl(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        self.unary_impl::<Neg>(shape, stride)
    }

    pub(crate) fn sqr_impl(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        self.unary_impl::<Sqr>(shape, stride)
    }

    pub(crate) fn sqrt_impl(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        self.unary_impl::<Sqrt>(shape, stride)
    }
}
