// src/benchmarks.rs

/// Basic fibonacci implementation for benchmarking
/// Note: In a real application, this would likely use u64 or bigger to handle larger numbers
#[ic_cdk::query]
fn fibonacci(n: u32) -> u32 {
    if n == 0 {
        return 0;
    } else if n == 1 {
        return 1;
    }

    let mut a = 0;
    let mut b = 1;
    let mut result = 0;

    for _ in 2..=n {
        result = a + b;
        a = b;
        b = result;
    }

    result
}

#[cfg(feature = "canbench-rs")]
mod computation_benchmarks {
    use super::*;
    use canbench_rs::bench;

    /// Base case benchmark
    /// Without #[inline(never)], all fibonacci benchmarks would show
    /// the same instruction count due to compiler optimization
    #[bench]
    fn fibonacci_base_case() {
        let _result = fibonacci(1);
    }

    /// Medium computation (n=20)
    #[bench]
    fn fibonacci_20() {
        let _result = fibonacci(20);
    }

    /// Large computation (n=45)
    #[bench]
    fn fibonacci_45() {
        let _result = fibonacci(45);
    }

    /// Pure loop implementation to compare with main function
    /// Must prevent inlining to stop compiler from optimizing away the loop
    #[inline(never)]
    fn fibonacci_loop(n: u32) -> u32 {
        let mut a = 0;
        let mut b = 1;
        let mut result = 0;

        for _ in 2..=n {
            result = a + b;
            a = b;
            b = result;
        }
        result
    }

    /// Loop-only benchmark for n=20
    /// The instruction count should scale with input size
    /// when optimizations are properly prevented
    #[bench]
    fn fibonacci_loop_20() {
        let _result = fibonacci_loop(20);
    }

    /// Loop-only benchmark for n=45
    /// Should show significantly more instructions than n=20
    /// due to more loop iterations
    #[bench]
    fn fibonacci_loop_45() {
        let _result = fibonacci_loop(45);
    }
}

#[cfg(feature = "canbench-rs")]
mod memory_benchmarks {
    use canbench_rs::bench;

    /// Tests heap allocation measurement with large vector
    /// Expected: ~61 pages for 1M integers (4MB)
    #[bench]
    fn benchmark_large_allocation() {
        let mut vec = Vec::with_capacity(1_000_000);
        for i in 0..1_000_000 {
            vec.push(i);
        }
        // Keep vec alive until end of benchmark
        assert!(vec.capacity() >= 1_000_000);
    }

    /// Tests heap measurement with retained allocation
    /// Expected: ~6 pages for 100K capacity (400KB)
    /// Note: Uses unsafe code to prevent deallocation
    static mut RETAINED_VEC: Option<Vec<i32>> = None;

    #[bench]
    fn benchmark_retained_allocation() {
        let vec = Vec::with_capacity(100_000);
        unsafe {
            RETAINED_VEC = Some(vec);
        }
    }

    /// Tests string allocation and growth
    /// Expected: ~46 pages for 1MB string
    #[bench]
    fn benchmark_string_allocation() {
        let mut big_string = String::with_capacity(1_000_000);
        for _ in 0..100_000 {
            big_string.push_str("test string ");
        }
        // Keep string alive
        assert!(big_string.capacity() >= 900_000);
    }
}