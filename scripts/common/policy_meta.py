from typing import Any, Dict, List


def flatten_policy_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}

    policy = meta.get("policy")
    if not isinstance(policy, dict):
        return {}

    out: Dict[str, Any] = {}
    if policy.get("page_ptr_id") is not None:
        out["iptp_id"] = str(policy.get("page_ptr_id"))

    title = policy.get("title") or policy.get("slug")
    if title is not None:
        out["title"] = str(title)

    if policy.get("announced_date") is not None:
        out["announced_date"] = str(policy.get("announced_date"))

    if policy.get("effective_date") is not None:
        out["effective_date"] = str(policy.get("effective_date"))

    if policy.get("administration") is not None:
        out["administration"] = str(policy.get("administration"))

    filters = policy.get("filters")
    if isinstance(filters, dict):
        agencies = filters.get("agencies")
        if isinstance(agencies, list):
            vals: List[str] = []
            for a in agencies:
                if not isinstance(a, dict):
                    continue
                v = a.get("agency") or a.get("agency_slug")
                if v:
                    vals.append(str(v))
            if vals:
                out["agencies_affected"] = vals
        subjects = filters.get("subject_matter")
        if isinstance(subjects, list):
            vals2: List[str] = []
            for s in subjects:
                if not isinstance(s, dict):
                    continue
                v = s.get("title") or s.get("slug")
                if v:
                    vals2.append(str(v))
            if vals2:
                out["subject_matter"] = vals2

    return out
