    def _thinker_decode_to_talker_decode(
        self,
        info_dict: dict,
        device: torch.device,
        update_dict,
    ):
        """
        Project thinker outputs to talker inputs during prefill stage.
        Returns:
            (input_ids, input_embeds) for talker
        """
        thinker_embed = info_dict.get("thinker_embeddings", None)
        if thinker_embed is None:
            if info_dict.get("finished_flag"):
                return self.tts_pad_embed.to(device)
            update_dict["finished_flag"] = True
            return self.tts_eos_embed.to(device)

        thinker_embed = thinker_embed.to(device)
        current_embed = thinker_embed[0:1]
        if thinker_embed.shape[0] > 1:
            update_dict["thinker_embeddings"] = thinker_embed[1:].detach().to("cpu").contiguous()
        else:
            update_dict["thinker_embeddings"] = None

        return self.talker.text_projection(current_embed).to(device)
