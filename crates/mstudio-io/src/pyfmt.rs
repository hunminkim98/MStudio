//! Python-compatible float formatting.
//!
//! The Python app writes TRC files through `pandas.to_csv`, which prints
//! floats with `repr()` (shortest round-trip digits, fixed notation for
//! exponents in `-4..16`, otherwise `1e-05` / `1.5e+20`). Rust's `{}` differs
//! (`1` for `1.0`, never scientific), so this reproduces Python's rules on
//! top of Rust's shortest-digit `{:e}` output to keep TRC writes byte-exact.

pub fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    // Shortest round-trip mantissa and decimal exponent, e.g. "-2.348508e-1".
    let sci = format!("{x:e}");
    let (mant, exp) = sci.split_once('e').expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();

    let mut s = String::with_capacity(digits.len() + 8);
    if neg {
        s.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() > int_len {
                s.push_str(&digits[..int_len]);
                s.push('.');
                s.push_str(&digits[int_len..]);
            } else {
                s.push_str(&digits);
                s.push_str(&"0".repeat(int_len - digits.len()));
                s.push_str(".0");
            }
        } else {
            s.push_str("0.");
            s.push_str(&"0".repeat((-exp - 1) as usize));
            s.push_str(&digits);
        }
    } else {
        s.push_str(&digits[..1]);
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push('e');
        s.push(if exp < 0 { '-' } else { '+' });
        s.push_str(&format!("{:02}", exp.abs()));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::py_repr;

    #[test]
    fn matches_python_repr() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (120.0, "120.0"),
            (100.0, "100.0"),
            (0.5, "0.5"),
            (9.5416667, "9.5416667"),
            (-0.2348508, "-0.2348508"),
            (12345.678, "12345.678"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.000123, "0.000123"),
            (0.0000123, "1.23e-05"),
            (1e16, "1e+16"),
            (1.5e20, "1.5e+20"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (999999999999999.9, "999999999999999.9"),
            (0.1 + 0.2, "0.30000000000000004"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ];
        for (x, want) in cases {
            assert_eq!(py_repr(*x), *want, "{x:?}");
        }
    }
}
