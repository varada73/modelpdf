---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:532761
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: what does the mineral name tell us
  sentences:
  - (United States). The average pay for a School Nurse is $39,318 per year. Pay for
    this job does not change much by experience, with the most experienced earning
    only a bit more than the least.A skill in Case Management is associated with high
    pay for this job.$23,589 - $61,194.he average pay for a School Nurse is $39,318
    per year. Pay for this job does not change much by experience, with the most experienced
    earning only a bit more than the least.
  - Owensville is a city in Gasconade County, Missouri, United States. The population
    was 2,676 at the 2010 census.
  - Amethyst, a variety of quartz. A mineral is a naturally occurring chemical compound,
    usually of crystalline form and abiogenic in origin. A mineral has one specific
    chemical composition, whereas a rock can be an aggregate of different minerals
    or mineraloids. The study of minerals is called mineralogy.
- source_sentence: what is the outer core composition
  sentences:
  - The outer core of the Earth is a liquid mix of elements, mostly iron and nickel,
    with smaller amounts of silicon and oxygen. It goes from a depth of 2550 km below
    the surface of the Earth to 4750 km at the boundary of the inner core, during
    which time the temperature increases from 4500 to 5500 degrees Celsius.
  - The Curse of the Billy Goat is a sports-related curse that was placed on the Chicago
    Cubs in 1945 when Billy Goat Tavern owner Billy Sianis was asked to leave a World
    Series game against the Detroit Tigers at the Cubs' home ballpark of Wrigley Field
    because the odor of his pet goat named Murphy was bothering other fans.
  - 'Definition. The genitive is the case (or function) of an inflected form of a
    noun or pronoun showing ownership, measurement, association, or source. Adjective:
    genitival. The suffix -''s on nouns is a marker of genitive case in English.'
- source_sentence: how did the treaty of tordesillas affect spain and portugal
  sentences:
  - The Americas. The Treaty of Tordesillas essentially divided the world between
    the two major imperial powers at the time, Spain and Portugal. Portugal received
    Africa, Asia, an…d what is modern-day eastern Brazil. Spain got the rest of the
    Americas.
  - Tulips are spring-blooming perennials that grow from bulbs. Depending on the species,
    tulip plants can be between 4 inches (10 cm) and 28 inches (71 cm) high. The tulip's
    flowers usually bloom on scapes with leaves in a rosette at ground level and a
    single flowering stalk arising from amongst the leaves.
  - Some THC metabolites have an elimination half-life of 20 hours. However, some
    are stored in body fat and have an elimination half-life of 10 to 13 days. Most
    researchers agree that urine tests for marijuana can detect the drug in the body
    for up to 13 days. However, there is anecdotal evidence that the length of time
    that marijuana remains in the body is affected by how often the person smokes,
    how much he smokes and how long he has been smoking.
- source_sentence: what is a colectomy
  sentences:
  - Colectomy is a surgical procedure to remove all or part of the colon. When only
    part of the colon is removed, it is called a partial colectomy.
  - In macOS, keychain files are stored in ~/Library/Keychains/, /Library/Keychains/,
    and /Network/Library/Keychains/, and the Keychain Access GUI application is located
    in the Utilities folder in the Applications folder. It is free, open source software
    released under the terms of the APSL.
  - annular eclipse-A solar eclipse in which the Moon's antumbral shadow traverses
    Earth (the Moon is too far from Earth to completely cover the Sun). During the
    maximum phase of an annular eclipse, the Sun appears as a blindingly bright ring
    surrounding the Moon.otality-The maximum phase of a total eclipse during which
    the Moon's disk completely covers the Sun. Totality is the period between second
    and third contact during a total eclipse. It can last from a fraction of a second
    to a maximum of 7 minutes 32 seconds.
- source_sentence: what age can a baby have blueberries
  sentences:
  - Berries are good sources of vitamin C. Once they are at least 6 months old and
    ready to eat solid foods, babies can eat most foods, including strawberries and
    blueberries, as long as they are in the proper form.Young babies need their fruits
    pureed, while babies at least 8 months old can eat fruit cut up into fingertip-sized
    pieces.ait for three days after first introducing strawberries or blueberries
    to your child to make sure he doesn't exhibit signs of an allergy, such as rash
    or swelling. If there is a family history of allergies, your child's doctor may
    recommend waiting until your baby is 1 year old before introducing strawberries.
  - 'Ableist. A PERSON or People who discriminate or social prejudice against people
    with disabilities. It can also be someone who judges or makes fun of someone with
    a disability or handicap. Example: Someone who makes fun/discriminates someone
    in a wheelchair or someone with down syndrome.'
  - The program provides a foundation for graduate education in nursing and serves
    as a stimulus for continuing professional development. Students who successfully
    complete the undergraduate BSN curriculum plan of studies (includes a Comprehension
    Exam) will be eligible to take the NCLEX to become RN’s. Registered nurses, who
    are graduates of diploma or associate degree programs in nursing, may choose to
    enroll in the RN Options.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'what age can a baby have blueberries',
    "Berries are good sources of vitamin C. Once they are at least 6 months old and ready to eat solid foods, babies can eat most foods, including strawberries and blueberries, as long as they are in the proper form.Young babies need their fruits pureed, while babies at least 8 months old can eat fruit cut up into fingertip-sized pieces.ait for three days after first introducing strawberries or blueberries to your child to make sure he doesn't exhibit signs of an allergy, such as rash or swelling. If there is a family history of allergies, your child's doctor may recommend waiting until your baby is 1 year old before introducing strawberries.",
    'Ableist. A PERSON or People who discriminate or social prejudice against people with disabilities. It can also be someone who judges or makes fun of someone with a disability or handicap. Example: Someone who makes fun/discriminates someone in a wheelchair or someone with down syndrome.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 532,761 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                          |
  |:--------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                              |
  | details | <ul><li>min: 4 tokens</li><li>mean: 8.88 tokens</li><li>max: 34 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 79.92 tokens</li><li>max: 197 tokens</li></ul> |
* Samples:
  | sentence_0                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
  |:------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>how long is it safe to keep butter out</code>         | <code>Butter will generally last for about one month after the sell-by date on the package, assuming it has been continuously refrigerated. How long can butter be left out at room temperature? Butter may be kept at room temperature for 1 to 2 days, but it will begin to spoil rapidly if not refrigerated after that. To further extend the shelf life of butter, freeze it: to freeze butter, wrap tightly in heavy-duty aluminum foil or plastic freezer wrap, or place inside a heavy-duty freezer bag.</code>                                                                                                                                        |
  | <code>is toxoplasma gondii a eukaryotic?</code>             | <code>Toxoplasma gondii is a single-celled eukaryotic protozoan parasite. The name Toxoplasma is derived from the shape of the organism, which is crescent-like (toxon is Greek for “arc”). T. gondii holds notoriety as the pathogen that causes the disease toxoplasmosis in humans.</code>                                                                                                                                                                                                                                                                                                                                                                  |
  | <code>what is the cytoplasm function in a plant cell</code> | <code>Best Answer: The cytoplasm function in a plant cell is almost similar to the cytoplasm function in an animal cell. In general, cytoplasm function in a cell is almost a mechanical one. It provides support to the internal structures by being a medium for their suspension.Cytoplasm function in a cell includes its role in maintaining the shape and consistency of the cell.n general, cytoplasm function in a cell is almost a mechanical one. It provides support to the internal structures by being a medium for their suspension. Cytoplasm function in a cell includes its role in maintaining the shape and consistency of the cell.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0601 | 500   | 0.0186        |
| 0.1201 | 1000  | 0.0167        |
| 0.1802 | 1500  | 0.0154        |
| 0.2402 | 2000  | 0.0159        |
| 0.3003 | 2500  | 0.016         |
| 0.3604 | 3000  | 0.0184        |
| 0.4204 | 3500  | 0.0169        |
| 0.4805 | 4000  | 0.0164        |
| 0.5405 | 4500  | 0.0179        |
| 0.6006 | 5000  | 0.0175        |
| 0.6607 | 5500  | 0.0156        |
| 0.7207 | 6000  | 0.015         |
| 0.7808 | 6500  | 0.017         |
| 0.8408 | 7000  | 0.0162        |
| 0.9009 | 7500  | 0.0173        |
| 0.9610 | 8000  | 0.0152        |
| 1.0210 | 8500  | 0.0154        |
| 1.0811 | 9000  | 0.0103        |
| 1.1411 | 9500  | 0.0105        |
| 1.2012 | 10000 | 0.0113        |
| 1.2613 | 10500 | 0.0117        |
| 1.3213 | 11000 | 0.0115        |
| 1.3814 | 11500 | 0.0121        |
| 1.4414 | 12000 | 0.0107        |
| 1.5015 | 12500 | 0.0107        |
| 1.5616 | 13000 | 0.0116        |
| 1.6216 | 13500 | 0.0111        |
| 1.6817 | 14000 | 0.0108        |
| 1.7417 | 14500 | 0.0119        |
| 1.8018 | 15000 | 0.0106        |
| 1.8619 | 15500 | 0.0111        |
| 1.9219 | 16000 | 0.0108        |
| 1.9820 | 16500 | 0.011         |
| 2.0420 | 17000 | 0.0091        |
| 2.1021 | 17500 | 0.0082        |
| 2.1622 | 18000 | 0.0078        |
| 2.2222 | 18500 | 0.0078        |
| 2.2823 | 19000 | 0.0097        |
| 2.3423 | 19500 | 0.0085        |
| 2.4024 | 20000 | 0.0079        |
| 2.4625 | 20500 | 0.0088        |
| 2.5225 | 21000 | 0.008         |
| 2.5826 | 21500 | 0.0078        |
| 2.6426 | 22000 | 0.0081        |
| 2.7027 | 22500 | 0.0085        |
| 2.7628 | 23000 | 0.0086        |
| 2.8228 | 23500 | 0.008         |
| 2.8829 | 24000 | 0.0078        |
| 2.9429 | 24500 | 0.0075        |


### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 4.1.0
- Transformers: 4.53.2
- PyTorch: 2.6.0+cu124
- Accelerate: 1.9.0
- Datasets: 2.14.4
- Tokenizers: 0.21.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->