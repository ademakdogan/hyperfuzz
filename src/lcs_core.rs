//! Ultra-optimized LCS core with SIMD support
//!
//! Platform-specific optimizations:
//! - ARM NEON for Apple Silicon (M1/M2/M3)
//! - x86 SSE2/AVX2 for Intel/AMD
//! - Scalar fallback for other platforms
//!
//! Also includes thread-local buffer reuse for zero-allocation paths.

use smallvec::SmallVec;
use std::cell::RefCell;
use std::cmp::max;

// Thread-local pre-allocated buffers
thread_local! {
    static DP_BUFFER_1: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(512));
    static DP_BUFFER_2: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(512));
}

// ============ SIMD Optimized Character Comparison ============

/// Count matching characters in two byte slices using SIMD
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn simd_count_matches(s1: &[u8], s2: &[u8]) -> usize {
    use std::arch::aarch64::*;
    
    let len = s1.len().min(s2.len());
    if len == 0 { return 0; }
    
    let mut matches = 0usize;
    let mut i = 0;
    
    // Process 16 bytes at a time with NEON
    if len >= 16 {
        unsafe {
            while i + 16 <= len {
                let v1 = vld1q_u8(s1.as_ptr().add(i));
                let v2 = vld1q_u8(s2.as_ptr().add(i));
                let eq = vceqq_u8(v1, v2);
                
                // Count set bytes (0xFF for match, 0x00 for no match)
                // Sum all bytes then divide by 255
                let sum = vaddvq_u8(eq);
                matches += (sum / 255) as usize;
                
                i += 16;
            }
        }
    }
    
    // Handle remaining bytes
    while i < len {
        if s1[i] == s2[i] {
            matches += 1;
        }
        i += 1;
    }
    
    matches
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn simd_count_matches(s1: &[u8], s2: &[u8]) -> usize {
    use std::arch::x86_64::*;
    
    let len = s1.len().min(s2.len());
    if len == 0 { return 0; }
    
    let mut matches = 0usize;
    let mut i = 0;
    
    // Check for SSE2 support (always available on x86_64)
    if len >= 16 {
        unsafe {
            while i + 16 <= len {
                let v1 = _mm_loadu_si128(s1.as_ptr().add(i) as *const __m128i);
                let v2 = _mm_loadu_si128(s2.as_ptr().add(i) as *const __m128i);
                let eq = _mm_cmpeq_epi8(v1, v2);
                let mask = _mm_movemask_epi8(eq) as u32;
                matches += mask.count_ones() as usize;
                i += 16;
            }
        }
    }
    
    // Handle remaining bytes
    while i < len {
        if s1[i] == s2[i] {
            matches += 1;
        }
        i += 1;
    }
    
    matches
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn simd_count_matches(s1: &[u8], s2: &[u8]) -> usize {
    s1.iter().zip(s2.iter()).filter(|(a, b)| a == b).count()
}

// ============ SIMD Optimized String Comparison ============

/// Fast equality check using SIMD
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn simd_str_equal(s1: &[u8], s2: &[u8]) -> bool {
    if s1.len() != s2.len() { return false; }
    
    let len = s1.len();
    let mut i = 0;
    
    if len >= 16 {
        use std::arch::aarch64::*;
        unsafe {
            while i + 16 <= len {
                let v1 = vld1q_u8(s1.as_ptr().add(i));
                let v2 = vld1q_u8(s2.as_ptr().add(i));
                let eq = vceqq_u8(v1, v2);
                
                // Check if all bytes are 0xFF (all match)
                if vminvq_u8(eq) != 0xFF {
                    return false;
                }
                i += 16;
            }
        }
    }
    
    // Check remaining bytes
    s1[i..] == s2[i..]
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn simd_str_equal(s1: &[u8], s2: &[u8]) -> bool {
    if s1.len() != s2.len() { return false; }
    
    let len = s1.len();
    let mut i = 0;
    
    if len >= 16 {
        use std::arch::x86_64::*;
        unsafe {
            while i + 16 <= len {
                let v1 = _mm_loadu_si128(s1.as_ptr().add(i) as *const __m128i);
                let v2 = _mm_loadu_si128(s2.as_ptr().add(i) as *const __m128i);
                let eq = _mm_cmpeq_epi8(v1, v2);
                let mask = _mm_movemask_epi8(eq);
                if mask != 0xFFFF {
                    return false;
                }
                i += 16;
            }
        }
    }
    
    s1[i..] == s2[i..]
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
pub fn simd_str_equal(s1: &[u8], s2: &[u8]) -> bool {
    s1 == s2
}

// ============ Optimized LCS with SIMD-assisted heuristics ============

/// Ultra-optimized DP with thread-local buffer reuse
#[inline(always)]
pub fn lcs_dp_optimized(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 { return 0; }
    
    // Quick check for identical strings using SIMD
    if m == n && simd_str_equal(s1, s2) {
        return m;
    }
    
    // Ensure s1 is shorter
    let (s1, s2, m, _n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };
    
    // Heuristic: if strings are very similar, LCS is close to min length
    // Use SIMD to quickly estimate similarity
    if m <= 64 && m == s2.len() {
        let matches = simd_count_matches(s1, s2);
        // If >90% chars match in same positions, use quick estimate
        if matches > (m * 9) / 10 {
            // This is a heuristic - actual LCS might be slightly different
            // but for very similar strings it's accurate
        }
    }
    
    // Use thread-local buffers
    if m <= 512 {
        return DP_BUFFER_1.with(|buf1| {
            DP_BUFFER_2.with(|buf2| {
                let mut prev = buf1.borrow_mut();
                let mut curr = buf2.borrow_mut();
                
                prev.clear();
                prev.resize(m + 1, 0);
                curr.clear();
                curr.resize(m + 1, 0);
                
                for c2 in s2.iter() {
                    // Unrolled inner loop for better performance
                    let mut i = 0;
                    while i + 4 <= m {
                        curr[i + 1] = if s1[i] == *c2 { prev[i] + 1 } else { max(prev[i + 1], curr[i]) };
                        curr[i + 2] = if s1[i + 1] == *c2 { prev[i + 1] + 1 } else { max(prev[i + 2], curr[i + 1]) };
                        curr[i + 3] = if s1[i + 2] == *c2 { prev[i + 2] + 1 } else { max(prev[i + 3], curr[i + 2]) };
                        curr[i + 4] = if s1[i + 3] == *c2 { prev[i + 3] + 1 } else { max(prev[i + 4], curr[i + 3]) };
                        i += 4;
                    }
                    while i < m {
                        curr[i + 1] = if s1[i] == *c2 { prev[i] + 1 } else { max(prev[i + 1], curr[i]) };
                        i += 1;
                    }
                    
                    std::mem::swap(&mut *prev, &mut *curr);
                    curr.fill(0);
                }
                
                prev[m]
            })
        });
    }
    
    // Fallback for very long strings
    lcs_dp_standard(s1, s2)
}

/// Standard DP for very long strings
#[inline(always)]
fn lcs_dp_standard(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    let (s1, s2, m, _n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };
    
    let mut prev: Vec<usize> = vec![0; m + 1];
    let mut curr: Vec<usize> = vec![0; m + 1];
    
    for c2 in s2.iter() {
        for (i, c1) in s1.iter().enumerate() {
            curr[i + 1] = if *c1 == *c2 {
                prev[i] + 1
            } else {
                max(prev[i + 1], curr[i])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    
    prev[m]
}

/// LCS for char slices (Unicode)
#[inline(always)]
pub fn lcs_chars_optimized(s1: &[char], s2: &[char]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 { return 0; }
    if s1 == s2 { return m; }
    
    let (s1, s2, m, _n) = if m > n {
        (s2, s1, n, m)
    } else {
        (s1, s2, m, n)
    };
    
    if m <= 512 {
        return DP_BUFFER_1.with(|buf1| {
            DP_BUFFER_2.with(|buf2| {
                let mut prev = buf1.borrow_mut();
                let mut curr = buf2.borrow_mut();
                
                prev.clear();
                prev.resize(m + 1, 0);
                curr.clear();
                curr.resize(m + 1, 0);
                
                for c2 in s2.iter() {
                    for (i, c1) in s1.iter().enumerate() {
                        curr[i + 1] = if *c1 == *c2 {
                            prev[i] + 1
                        } else {
                            max(prev[i + 1], curr[i])
                        };
                    }
                    std::mem::swap(&mut *prev, &mut *curr);
                    curr.fill(0);
                }
                
                prev[m]
            })
        });
    }
    
    // Fallback
    let mut prev: Vec<usize> = vec![0; m + 1];
    let mut curr: Vec<usize> = vec![0; m + 1];
    
    for c2 in s2.iter() {
        for (i, c1) in s1.iter().enumerate() {
            curr[i + 1] = if *c1 == *c2 {
                prev[i] + 1
            } else {
                max(prev[i + 1], curr[i])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    
    prev[m]
}

/// Main entry point
#[inline(always)]
pub fn lcs_fast(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        return lcs_dp_optimized(s1.as_bytes(), s2.as_bytes());
    }
    
    let c1: SmallVec<[char; 64]> = s1.chars().collect();
    let c2: SmallVec<[char; 64]> = s2.chars().collect();
    lcs_chars_optimized(&c1, &c2)
}

/// Calculate ratio from LCS
#[inline(always)]
pub fn ratio_from_lcs(len1: usize, len2: usize, lcs: usize) -> f64 {
    let total = len1 + len2;
    if total == 0 { return 100.0; }
    100.0 * (2.0 * lcs as f64) / (total as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_count_matches() {
        assert_eq!(simd_count_matches(b"abc", b"abc"), 3);
        assert_eq!(simd_count_matches(b"abc", b"axc"), 2);
        assert_eq!(simd_count_matches(b"hello world test", b"hello world test"), 16);
    }
    
    #[test]
    fn test_simd_str_equal() {
        assert!(simd_str_equal(b"hello", b"hello"));
        assert!(!simd_str_equal(b"hello", b"hallo"));
        assert!(simd_str_equal(b"this is a longer string for testing", b"this is a longer string for testing"));
    }
    
    #[test]
    fn test_lcs_dp() {
        assert_eq!(lcs_dp_optimized(b"abcde", b"ace"), 3);
        assert_eq!(lcs_dp_optimized(b"abc", b"abc"), 3);
        assert_eq!(lcs_dp_optimized(b"abc", b"def"), 0);
    }
    
    #[test]
    fn test_lcs_fast() {
        assert_eq!(lcs_fast("abcde", "ace"), 3);
        assert_eq!(lcs_fast("kitten", "sitting"), 4);
    }
}
