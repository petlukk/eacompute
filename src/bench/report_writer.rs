use super::*;

// --- ISO 8601 (Howard Hinnant's date algorithm; epoch is unix) ---

pub fn iso8601_utc(unix_secs: u64) -> String {
    let total = unix_secs as i64;
    let days = total.div_euclid(86_400);
    let secs_today = total.rem_euclid(86_400) as u32;
    let hour = secs_today / 3600;
    let min = (secs_today % 3600) / 60;
    let sec = secs_today % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y_base = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y_base + 1 } else { y_base };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

// --- JSON writer (string-builder, hand-rolled) ---

pub(super) fn esc(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn write_result_json(r: &Result_) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"schema_version\": ");
    s.push_str(&r.schema_version.to_string());
    s.push_str(",\n  \"name\": ");
    esc(&r.name, &mut s);
    s.push_str(",\n  \"eacompute_version\": ");
    esc(&r.eacompute_version, &mut s);
    s.push_str(",\n  \"git_sha\": ");
    match &r.git_sha {
        Some(g) => esc(g, &mut s),
        None => s.push_str("null"),
    }
    s.push_str(",\n  \"timestamp\": ");
    esc(&r.timestamp, &mut s);
    s.push_str(",\n  \"env\": {\n");
    s.push_str("    \"os\": ");
    esc(&r.env.os, &mut s);
    s.push_str(",\n    \"arch\": ");
    esc(&r.env.arch, &mut s);
    s.push_str(",\n    \"host_cpu\": ");
    esc(&r.env.host_cpu, &mut s);
    s.push_str(",\n    \"target_cpu\": ");
    esc(&r.env.target_cpu, &mut s);
    s.push_str(",\n    \"target_features\": ");
    esc(&r.env.target_features, &mut s);
    s.push_str(",\n    \"opt_level\": ");
    s.push_str(&r.env.opt_level.to_string());
    s.push_str(",\n    \"pinned\": ");
    s.push_str(if r.env.pinned { "true" } else { "false" });
    s.push_str("\n  },\n");
    s.push_str("  \"measurements\": [");
    for (i, m) in r.measurements.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("\n    {");
        s.push_str("\"kernel\": ");
        esc(&m.kernel, &mut s);
        s.push_str(", \"median_ns\": ");
        s.push_str(&m.median_ns.to_string());
        if let Some(v) = m.p10_ns {
            s.push_str(", \"p10_ns\": ");
            s.push_str(&v.to_string());
        }
        if let Some(v) = m.p90_ns {
            s.push_str(", \"p90_ns\": ");
            s.push_str(&v.to_string());
        }
        if let Some(v) = m.n_inner {
            s.push_str(", \"n_inner\": ");
            s.push_str(&v.to_string());
        }
        if let Some(v) = m.n_runs {
            s.push_str(", \"n_runs\": ");
            s.push_str(&v.to_string());
        }
        for (k, raw_val) in &m.extra {
            s.push_str(", ");
            esc(k, &mut s);
            s.push_str(": ");
            s.push_str(raw_val);
        }
        s.push('}');
    }
    s.push_str("\n  ]\n}\n");
    s
}
