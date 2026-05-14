use super::{REGRESSION_THRESHOLD, Result_};

#[derive(Debug, PartialEq)]
pub enum KernelStatus {
    Ok,
    Regression,
    Improvement,
    MissingFromCurrent,
    NewKernel,
}

#[derive(Debug)]
pub struct KernelDiff {
    pub kernel: String,
    pub current_ns: Option<u64>,
    pub baseline_ns: Option<u64>,
    pub delta_pct: Option<f64>,
    pub status: KernelStatus,
}

#[derive(Debug)]
pub struct DiffReport {
    pub kernels: Vec<KernelDiff>,
    pub regressions: u32,
    pub stale: bool,
    pub stale_reason: Option<String>,
}

pub fn diff(current: &Result_, baseline: &Result_) -> DiffReport {
    let mut stale_reasons = Vec::new();
    if current.env.arch != baseline.env.arch {
        stale_reasons.push(format!(
            "arch={} != current {}",
            baseline.env.arch, current.env.arch
        ));
    }
    if current.env.target_features != baseline.env.target_features {
        stale_reasons.push(format!(
            "target_features='{}' != current '{}'",
            baseline.env.target_features, current.env.target_features
        ));
    }
    if current.env.opt_level != baseline.env.opt_level {
        stale_reasons.push(format!(
            "opt_level={} != current {}",
            baseline.env.opt_level, current.env.opt_level
        ));
    }
    let stale = !stale_reasons.is_empty();
    let stale_reason = if stale {
        Some(stale_reasons.join("; "))
    } else {
        None
    };

    let mut kernels = Vec::new();
    let mut regressions = 0u32;

    for c in &current.measurements {
        let b = baseline.measurements.iter().find(|b| b.kernel == c.kernel);
        match b {
            Some(b) => {
                let delta = (c.median_ns as f64 - b.median_ns as f64) / b.median_ns as f64;
                let status = if delta > REGRESSION_THRESHOLD {
                    regressions += 1;
                    KernelStatus::Regression
                } else if delta < 0.0 {
                    KernelStatus::Improvement
                } else {
                    KernelStatus::Ok
                };
                kernels.push(KernelDiff {
                    kernel: c.kernel.clone(),
                    current_ns: Some(c.median_ns),
                    baseline_ns: Some(b.median_ns),
                    delta_pct: Some(delta * 100.0),
                    status,
                });
            }
            None => kernels.push(KernelDiff {
                kernel: c.kernel.clone(),
                current_ns: Some(c.median_ns),
                baseline_ns: None,
                delta_pct: None,
                status: KernelStatus::NewKernel,
            }),
        }
    }
    for b in &baseline.measurements {
        if !current.measurements.iter().any(|c| c.kernel == b.kernel) {
            kernels.push(KernelDiff {
                kernel: b.kernel.clone(),
                current_ns: None,
                baseline_ns: Some(b.median_ns),
                delta_pct: None,
                status: KernelStatus::MissingFromCurrent,
            });
        }
    }

    DiffReport {
        kernels,
        regressions,
        stale,
        stale_reason,
    }
}

pub fn format_diff(d: &DiffReport, current: &Result_) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{} ({}, {}, opt={}):\n",
        current.name, current.env.arch, current.env.target_cpu, current.env.opt_level
    ));
    for k in &d.kernels {
        match k.status {
            KernelStatus::MissingFromCurrent => {
                s.push_str(&format!(
                    "  {:<16} MISSING from current run (baseline {} ns)\n",
                    k.kernel,
                    k.baseline_ns.unwrap()
                ));
            }
            KernelStatus::NewKernel => {
                s.push_str(&format!(
                    "  {:<16} {:>10} ns  NEW kernel (no baseline)\n",
                    k.kernel,
                    k.current_ns.unwrap()
                ));
            }
            _ => {
                let delta = k.delta_pct.unwrap();
                let sign = if delta >= 0.0 { "+" } else { "" };
                s.push_str(&format!(
                    "  {:<16} {:>10} ns  (baseline {} ns, {}{:.1}%)\n",
                    k.kernel,
                    k.current_ns.unwrap(),
                    k.baseline_ns.unwrap(),
                    sign,
                    delta
                ));
            }
        }
    }
    s.push_str(&format!(
        "WARNING: {} regressions exceed {:.0}% threshold (warn-only in v1.13.0).\n",
        d.regressions,
        REGRESSION_THRESHOLD * 100.0
    ));
    s
}
