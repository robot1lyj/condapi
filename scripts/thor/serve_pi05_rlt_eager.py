"""Opt-in RLT experimental service, sharing the existing Pi/RTC server."""

import logging

from serve_pi05_parts_eager import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(
        manifest_option="--rlt-manifest",
        backend="eager_rlt_candidate",
        expected_feature_extractor="pi_eager_rlt_final_prefix_v1",
    )
