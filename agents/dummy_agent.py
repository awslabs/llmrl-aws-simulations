import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from typing import Optional, Union
from agents.linear_scheduler import LinearScheduler


class DummyAgent(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        accelerator: Accelerator,
        policy_lm: str = "gpt2",
        cache_dir: str = "~/.cache",
        template: Optional[str] = None,
        use_lora: bool = False,
        do_sample: bool = True,
        temperature: Union[float, LinearScheduler] = 1.0,
        eval_temperature: Union[float, LinearScheduler] = 0.1,
        max_new_tokens: int = 32,
        max_length_tokenizer: int = 512,
        use_bfloat16: bool = False,
        eos_str: str = "\n",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        quantization: Optional[int] = None,
        use_better_transformer: bool = False,
        clip_generation_with_stop_token: bool = True,
        attn_implementation: str = "sdpa",
        beam_size: int = 1,
    ):
        if isinstance(temperature, float):
            temperature = LinearScheduler(
                start=temperature,
                end=temperature,
                total_steps=1,
            )
        if isinstance(eval_temperature, float):
            eval_temperature = LinearScheduler(
                start=eval_temperature,
                end=eval_temperature,
                total_steps=1,
            )
        super(DummyAgent, self).__init__()
        pretrained_kwargs = {}
        self.quantization = False
        pretrained_kwargs["attn_implementation"] = attn_implementation
        if quantization:
            self.quantization = True
            pretrained_kwargs["device_map"] = {"": accelerator.process_index}
            if quantization == 8:
                pretrained_kwargs["load_in_8bit"] = True
            elif quantization == 4:
                pretrained_kwargs["load_in_4bit"] = True
            else:
                raise NotImplementedError
        self.pretrained_kwargs = pretrained_kwargs
        self.lora_kwargs = {
            "r": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }
        self.use_better_transformer = use_better_transformer
        self.max_length_tokenizer = max_length_tokenizer
        self.policy_lm = policy_lm
        self.cache_dir = cache_dir
        self.use_bfloat16 = use_bfloat16
        self.use_lora = use_lora
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str
        self.template = template
        self.device = device

        self.model = self.setup_model()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.policy_lm, trust_remote_code=True
        )
        # specify the side from which we should truncate the tokens --> left to right, i.e., natural language
        self.tokenizer.truncation_side = "left"
        # specify pad token to be toe eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.softmax = torch.nn.Softmax(dim=-1)
        self.do_sample = do_sample
        self._temperature = temperature
        self.accelerator = accelerator
        self._evaluation = False
        self._eval_temperature = eval_temperature

        self.anchor = None
        self.stop_token_id = self.tokenizer(self.eos_str)["input_ids"][0]
        self.clip_generation_with_stop_token = clip_generation_with_stop_token
        self.beam_size = beam_size

    def setup_model(self):
        # can only use bfloat 16 or quantization
        if self.use_bfloat16 and not self.quantization:
            model = AutoModelForCausalLM.from_pretrained(
                self.policy_lm,
                cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16,
                **self.pretrained_kwargs,
            ).to(self.device)
        else:
            # mapping to device is automatically done when using quantization
            model = AutoModelForCausalLM.from_pretrained(
                self.policy_lm, cache_dir=self.cache_dir, **self.pretrained_kwargs
            )
            if not self.quantization:
                model = model.to(self.device)
        if self.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                target_modules="all-linear",
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
                **self.lora_kwargs,
            )
            model = get_peft_model(model, lora_config)
            print("Using LoRA")
            model.print_trainable_parameters()
        if self.use_better_transformer:
            model = model.to_bettertransformer()
        return model

    def set_policy_to_eval_mode(self):
        self._evaluation = True

    def set_policy_to_train_mode(self):
        self._evaluation = False

    @property
    def has_anchor(self):
        return self.anchor is not None

    @property
    def temperature(self):
        return self._temperature()

    @property
    def eval_temperature(self):
        return self._eval_temperature()

    def update_temperature(self):
        self._temperature.update()
        self._eval_temperature.update()

    def setup_anchor(self, anchor_checkpoint: str, map_location=None):
        # setup a model similar to the anchor and load its state dict
        anchor = self.setup_model()
        checkpoint = torch.load(anchor_checkpoint, map_location=map_location)
        anchor.load_state_dict(checkpoint["model_state_dict"])
        self.anchor = anchor

    def get_model_state_dict(self):
        model = self.accelerator.unwrap_model(self.model)
        if self.use_better_transformer:
            model = model.reverse_bettertransformer()
        return {
            "model_state_dict": model.state_dict(),
            "temperature_state_dict": {
                "train": self._temperature.state_dict(),
                "eval": self._eval_temperature.state_dict(),
            },
        }

    def load_temperature_from_state_dict(self, state_dict):
        self._temperature.load_from_state_dict(state_dict["train"])
        self._eval_temperature.load_from_state_dict(state_dict["eval"])

    def prepare(self):
        self.model = self.accelerator.prepare(self.model)
        if self.has_anchor:
            self.anchor = self.accelerator.prepare(self.anchor)

    def get_model_from_model_type(self, model_type: str = "policy"):
        assert model_type in ["policy", "anchor"]
        if model_type == "anchor":
            assert (
                self.has_anchor
            ), "Cannot use anchor model since it is not initialized."
            model = self.anchor
        else:
            model = self.model
        return model

    def get_action_for_model(self, observation, model_type: str = "policy"):
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        model = self.get_model_from_model_type(model_type=model_type)
        # tokenize input, truncate it if its longer than max_length_tokenizer
        obs_ids = self.tokenizer(
            observation,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length_tokenizer,
            truncation=True,
        ).to(self.device)
        context_len = obs_ids["attention_mask"].size(1)
        if self._evaluation:
            temperature = self.eval_temperature
            if self.do_sample and self.beam_size > 1:
                generation_kwargs = {
                    'num_beams': self.beam_size,
                }
            else:
                generation_kwargs = {}
        else:
            temperature = self.temperature
            generation_kwargs = {}
        # when using this, we stop the generation process as soon as the stop token id is outputted.
        if self.clip_generation_with_stop_token:
            # taken from here: https://github.com/huggingface/transformers/issues/23175
            outputs = (
                self.accelerator.unwrap_model(model)
                .generate(
                    **obs_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.stop_token_id,
                    **generation_kwargs
                )
                .cpu()
            )
        else:
            outputs = (
                self.accelerator.unwrap_model(model)
                .generate(
                    **obs_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
                .cpu()
            )

        # removed the obs from the output
        outputs = outputs[:, context_len:]
        # except:
        #     import IPython; IPython.embed()
        # decode the generated outputs
        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # remove begining \n
        # If the first 3 starting tokens are '\n', remove them from the action.
        for _ in range(3):
            raw_action = [a[1:] if a.startswith("\n") else a for a in raw_action]
        # Remove the padded outputs (eos_str) from the raw outputs.
        if self.eos_str is not None:
            # print(f"using eos str {eos_str}")
            # print([raw_a.split(self.eos_str)[0] + self.eos_str for raw_a in raw_action])
            return [raw_a.split(self.eos_str)[0] + self.eos_str for raw_a in raw_action]
        else:
            return raw_action

    def _get_action(self, observation):
        with torch.no_grad():
            return self.get_action_for_model(observation, model_type="policy")

    def get_action(self, observation, **kwargs):
        raise NotImplementedError

    def get_log_prob_for_model(self, observation, action, model_type: str = "policy"):
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        model = self.get_model_from_model_type(model_type=model_type)
        # tokenize observations. Truncate obs if its greater than max_length_tokenizer
        obs_ids = self.tokenizer(
            observation,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length_tokenizer,
            truncation=True,
        ).to(self.device)

        # tokenize actions. Truncation action if its greater than max_length_tokenizer
        action_ids = self.tokenizer(
            action,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length_tokenizer,
            truncation=True,
        ).to(self.device)
        # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        # Take inputs and their corresponding attentions and pass it through the model to get the logits.
        input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim=1)
        # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        attention_mask = torch.cat(
            [obs_ids["attention_mask"], action_ids["attention_mask"]], dim=1
        )
        # pass the inputs and attention mask to get outputs. Attention mask tells which tokens should be attended to.
        # For example, padded tokens are ignored-
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # prediction probs: [B x N x V], where V: Number of tokens in the vocabulary
        prediction_probs = self.softmax(outputs.logits)
        # Get probability of the action token which was passed, i.e., p(a|s) by select the token a from V
        # model_output gives us p(x_t+1|x<=t) at index t
        # given this when we have observations of sequence L. We get at index L p(x_L+1|x<=L)
        # Following this logic, the action starts at index L.
        selected_prediction_probs = torch.take_along_dim(
            prediction_probs[:, obs_ids["attention_mask"].size(1) - 1 : -1],
            action_ids["input_ids"].unsqueeze(2),
            dim=2,
        ).squeeze(2)
        # mask out the probabilities that are not attended to/tokens that were generated due to padding.
        logsum_probs = torch.sum(
            torch.log(selected_prediction_probs) * action_ids["attention_mask"], dim=1
        )
        return logsum_probs

    def get_log_prob(self, observation, action):
        return self.get_log_prob_for_model(observation, action, model_type="policy")
