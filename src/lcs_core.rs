//! Ultra-optimized LCS core with bit-parallel algorithm
//!
//! Implements Hyyrö's bit-parallel algorithm for O(n*m/64) complexity
//! Reference: Heikki Hyyrö - "A Note on Bit-Parallel Alignment Computation" (2004)
//!
//! Also includes:
//! - SIMD-accelerated string comparison
//! - Thread-local buffer reuse
//! - Character filtering for partial_ratio

use smallvec::SmallVec;
use std::cell::RefCell;
use std::cmp::max;
use ahash::AHashMap;

// Thread-local pre-allocated buffers
thread_local! {
    static DP_BUFFER_1: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(512));
    static DP_BUFFER_2: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(512));
    static CHAR_BLOCK: RefCell<AHashMap<u8, u64>> = RefCell::new(AHashMap::with_capacity(64));
}

// ============ Bit-Parallel LCS (Hyyrö's Algorithm) ============

/// Compute LCS length using bit-parallel algorithm for ASCII strings <= 64 chars
/// This is O(n) for fixed s1 length, O(n*m/64) overall
#[inline(always)]
pub fn lcs_bitparallel_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    if len1 == 0 { return 0; }
    if len1 > 64 { return lcs_dp_optimized(s1, s2); }
    
    // Build character block map
    let mut block = [0u64; 256];
    for (i, &c) in s1.iter().enumerate() {
        block[c as usize] |= 1u64 << i;
    }
    
    // Bit-parallel computation
    let mut s: u64 = (1u64 << len1) - 1;
    
    for &c2 in s2.iter() {
        let matches = block[c2 as usize];
        let u = s & matches;
        s = (s.wrapping_add(u)) | (s.wrapping_sub(u));
    }
    
    // Count zeros in the rightmost len1 bits = LCS length
    let mask = (1u64 << len1) - 1;
    len1 - (s & mask).count_ones() as usize
}

/// Multi-block bit-parallel LCS for strings > 64 characters
/// Uses multiple 64-bit words to handle longer strings
#[inline(always)]
pub fn lcs_bitparallel_multiblock(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    
    if len1 == 0 || len2 == 0 { return 0; }
    if len1 <= 64 { return lcs_bitparallel_ascii(s1, s2); }
    
    // Use specialized 2-block version for 65-128 chars (stack allocated)
    if len1 <= 128 {
        return lcs_bitparallel_2block(s1, s2);
    }
    
    // General multi-block for > 128 chars
    let num_blocks = (len1 + 63) / 64;
    let mut block_maps: Vec<[u64; 256]> = vec![[0u64; 256]; num_blocks];
    
    for (i, &c) in s1.iter().enumerate() {
        let block_idx = i / 64;
        let bit_pos = i % 64;
        block_maps[block_idx][c as usize] |= 1u64 << bit_pos;
    }
    
    let mut s_vec: Vec<u64> = vec![!0u64; num_blocks];
    
    for &c2 in s2.iter() {
        let mut carry: u64 = 0;
        
        for block_idx in 0..num_blocks {
            let matches = block_maps[block_idx][c2 as usize];
            let s_old = s_vec[block_idx];
            
            let u = s_old & matches;
            let (sum, overflow1) = s_old.overflowing_add(u);
            let sum_with_carry = sum.wrapping_add(carry);
            let overflow2 = sum_with_carry < sum;
            
            s_vec[block_idx] = sum_with_carry | (s_old.wrapping_sub(u));
            carry = (overflow1 || overflow2) as u64;
        }
    }
    
    let mut lcs = 0usize;
    for (block_idx, &s) in s_vec.iter().enumerate() {
        let block_start = block_idx * 64;
        let block_len = std::cmp::min(64, len1 - block_start);
        let mask = if block_len == 64 { !0u64 } else { (1u64 << block_len) - 1 };
        lcs += block_len - (s & mask).count_ones() as usize;
    }
    
    lcs
}

/// Specialized 2-block LCS for strings 65-128 chars (full stack allocation)
#[inline(always)]
fn lcs_bitparallel_2block(s1: &[u8], s2: &[u8]) -> usize {
    let len1 = s1.len();
    
    // Stack-allocated block maps (2 blocks × 256 entries)
    let mut block0: [u64; 256] = [0u64; 256];
    let mut block1: [u64; 256] = [0u64; 256];
    
    for (i, &c) in s1.iter().enumerate() {
        if i < 64 {
            block0[c as usize] |= 1u64 << i;
        } else {
            block1[c as usize] |= 1u64 << (i - 64);
        }
    }
    
    // Stack-allocated state (2 × u64)
    let mut s0: u64 = !0u64;
    let mut s1_state: u64 = !0u64;
    
    for &c2 in s2.iter() {
        let matches0 = block0[c2 as usize];
        let matches1 = block1[c2 as usize];
        
        // Block 0
        let u0 = s0 & matches0;
        let (sum0, overflow0) = s0.overflowing_add(u0);
        s0 = sum0 | (s0.wrapping_sub(u0));
        let carry = overflow0 as u64;
        
        // Block 1
        let u1 = s1_state & matches1;
        let (sum1, _) = s1_state.overflowing_add(u1);
        let sum1_carry = sum1.wrapping_add(carry);
        s1_state = sum1_carry | (s1_state.wrapping_sub(u1));
    }
    
    // Count LCS
    let len_block0 = std::cmp::min(64, len1);
    let len_block1 = len1 - 64;
    
    let mask0 = !0u64; // full 64 bits
    let mask1 = if len_block1 == 64 { !0u64 } else { (1u64 << len_block1) - 1 };
    
    let lcs0 = 64 - (s0 & mask0).count_ones() as usize;
    let lcs1 = len_block1 - (s1_state & mask1).count_ones() as usize;
    
    lcs0 + lcs1
}

/// Compute LCS using pre-built block map (for sliding window operations)
#[inline(always)]
pub fn lcs_with_block(block: &[u64; 256], len1: usize, s2: &[u8]) -> usize {
    if len1 == 0 { return 0; }
    
    let mut s: u64 = (1u64 << len1) - 1;
    
    for &c2 in s2.iter() {
        let matches = block[c2 as usize];
        let u = s & matches;
        s = (s.wrapping_add(u)) | (s.wrapping_sub(u));
    }
    
    let mask = (1u64 << len1) - 1;
    len1 - (s & mask).count_ones() as usize
}

/// Build character block map for bit-parallel algorithm
#[inline(always)]
pub fn build_block_map(s1: &[u8]) -> [u64; 256] {
    let mut block = [0u64; 256];
    for (i, &c) in s1.iter().enumerate() {
        if i >= 64 { break; }
        block[c as usize] |= 1u64 << i;
    }
    block
}

/// Build character set for filtering
#[inline(always)]
pub fn build_char_set(s1: &[u8]) -> [bool; 256] {
    let mut char_set = [false; 256];
    for &c in s1.iter() {
        char_set[c as usize] = true;
    }
    char_set
}

// ============ SIMD Optimized String Comparison ============

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
                if vminvq_u8(eq) != 0xFF { return false; }
                i += 16;
            }
        }
    }
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
                if _mm_movemask_epi8(eq) != 0xFFFF { return false; }
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

// ============ Optimized DP LCS ============

/// Ultra-optimized DP with thread-local buffer reuse
#[inline(always)]
pub fn lcs_dp_optimized(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 { return 0; }
    if m == n && simd_str_equal(s1, s2) { return m; }
    
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
    let mut prev: Vec<usize> = vec![0; m + 1];
    let mut curr: Vec<usize> = vec![0; m + 1];
    for c2 in s2.iter() {
        for (i, c1) in s1.iter().enumerate() {
            curr[i + 1] = if *c1 == *c2 { prev[i] + 1 } else { max(prev[i + 1], curr[i]) };
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
                        curr[i + 1] = if *c1 == *c2 { prev[i] + 1 } else { max(prev[i + 1], curr[i]) };
                    }
                    std::mem::swap(&mut *prev, &mut *curr);
                    curr.fill(0);
                }
                prev[m]
            })
        });
    }
    
    let mut prev: Vec<usize> = vec![0; m + 1];
    let mut curr: Vec<usize> = vec![0; m + 1];
    for c2 in s2.iter() {
        for (i, c1) in s1.iter().enumerate() {
            curr[i + 1] = if *c1 == *c2 { prev[i] + 1 } else { max(prev[i + 1], curr[i]) };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    prev[m]
}

/// Main entry point - uses bit-parallel for all ASCII strings
#[inline(always)]
pub fn lcs_fast(s1: &str, s2: &str) -> usize {
    if s1.is_ascii() && s2.is_ascii() {
        let b1 = s1.as_bytes();
        let b2 = s2.as_bytes();
        
        // Use bit-parallel for all ASCII strings (multi-block handles > 64)
        return lcs_bitparallel_multiblock(b1, b2);
    }
    
    let c1: SmallVec<[char; 64]> = s1.chars().collect();
    let c2: SmallVec<[char; 64]> = s2.chars().collect();
    lcs_chars_optimized(&c1, &c2)
}

// ============ Longest Common Substring (LCSstr) ============
// Unlike LCSseq, this finds contiguous common substrings

/// Compute longest common substring length for ASCII strings
/// Uses suffix-based DP approach with O(min(m,n)) space
#[inline(always)]
pub fn lcsstr_ascii(s1: &[u8], s2: &[u8]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 {
        return 0;
    }
    
    // Ensure s1 is the shorter string for space optimization
    let (short, long) = if m <= n { (s1, s2) } else { (s2, s1) };
    let short_len = short.len();
    let long_len = long.len();
    
    // Single row DP - dp[j] = length of common substring ending at positions (i, j)
    let mut prev = vec![0usize; short_len + 1];
    let mut curr = vec![0usize; short_len + 1];
    let mut max_len = 0;
    
    for i in 1..=long_len {
        for j in 1..=short_len {
            if long[i - 1] == short[j - 1] {
                curr[j] = prev[j - 1] + 1;
                if curr[j] > max_len {
                    max_len = curr[j];
                }
            } else {
                curr[j] = 0;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    
    max_len
}

/// Compute longest common substring length for Unicode strings
#[inline(always)]
pub fn lcsstr_chars(s1: &[char], s2: &[char]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 {
        return 0;
    }
    
    // Ensure s1 is the shorter string
    let (short, long) = if m <= n { (s1, s2) } else { (s2, s1) };
    let short_len = short.len();
    let long_len = long.len();
    
    let mut prev = vec![0usize; short_len + 1];
    let mut curr = vec![0usize; short_len + 1];
    let mut max_len = 0;
    
    for i in 1..=long_len {
        for j in 1..=short_len {
            if long[i - 1] == short[j - 1] {
                curr[j] = prev[j - 1] + 1;
                if curr[j] > max_len {
                    max_len = curr[j];
                }
            } else {
                curr[j] = 0;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    
    max_len
}

/// Main entry point for LCSstr - uses optimized path for ASCII
#[inline(always)]
pub fn lcsstr_fast(s1: &str, s2: &str) -> usize {
    if s1.is_empty() || s2.is_empty() {
        return 0;
    }
    
    // Fast path for ASCII
    if s1.is_ascii() && s2.is_ascii() {
        return lcsstr_ascii(s1.as_bytes(), s2.as_bytes());
    }
    
    // Unicode path
    let c1: SmallVec<[char; 64]> = s1.chars().collect();
    let c2: SmallVec<[char; 64]> = s2.chars().collect();
    lcsstr_chars(&c1, &c2)
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
    fn test_lcs_bitparallel() {
        assert_eq!(lcs_bitparallel_ascii(b"abcde", b"ace"), 3);
        assert_eq!(lcs_bitparallel_ascii(b"abc", b"abc"), 3);
        assert_eq!(lcs_bitparallel_ascii(b"abc", b"def"), 0);
        assert_eq!(lcs_bitparallel_ascii(b"kitten", b"sitting"), 4);
    }
    
    #[test]
    fn test_lcs_with_block() {
        let block = build_block_map(b"abcde");
        assert_eq!(lcs_with_block(&block, 5, b"ace"), 3);
    }
    
    #[test]
    fn test_lcs_fast() {
        assert_eq!(lcs_fast("abcde", "ace"), 3);
        assert_eq!(lcs_fast("kitten", "sitting"), 4);
    }
}
