    def _get_additional_information(self, scheduler_output: "SchedulerOutput", req_id: str) -> dict:
        req_infos = None
        req_state = self.requests.get(req_id)
        additional_information_cpu = getattr(req_state, "additional_information_cpu", None)
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id == req_id:
                payload_info = getattr(new_req, "additional_information", None)
                if payload_info is not None:
                    return payload_info

        if hasattr(scheduler_output.scheduled_cached_reqs, "additional_information"):
            cached_infos = getattr(scheduler_output.scheduled_cached_reqs, "additional_information", {})
            if isinstance(cached_infos, dict) and req_id in cached_infos:
                req_infos = cached_infos[req_id]
                if not isinstance(req_infos, dict):
                    req_infos = None

        if req_infos is None or req_infos.get("last_talker_hidden", None) is None:
            if req_infos is None:
                # Do not pop here, let preprocess handle it.
                req_infos = additional_information_cpu
            else:
                if additional_information_cpu is not None:
                    req_infos["last_talker_hidden"] = additional_information_cpu.get("last_talker_hidden", None)
                    req_infos["num_processed_thinker_tokens"] = additional_information_cpu.get(
                        "num_processed_thinker_tokens", 0
                    )
            if not isinstance(req_infos, dict):
                req_infos = None

        if req_infos is None:
            logger.warning(f"No additional_information found for req_id: {req_id}")

        return req_infos
