        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                )
                if request:
                    request.additional_information = None
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
            cached_reqs = scheduler_output.scheduled_cached_reqs
            if not hasattr(cached_reqs, "additional_information"):
                cached_reqs.additional_information = {}
            for req_id in cached_reqs.req_ids:
                request = self.requests.get(req_id) if req_id else None
                additional_info = getattr(request, "additional_information", None) if request else None
                cached_reqs.additional_information[req_id] = additional_info
                if request:
                    request.additional_information = None

            # Add information about requests needing KV cache transfer
            scheduler_output.finished_requests_needing_kv_transfer = self.get_finished_requests_needing_kv_transfer()
        except Exception:
