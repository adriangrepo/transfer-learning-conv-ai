Added tokenization_gpt2.py and file_utils.py from pytorch-retrained-BERT rather than import so can trace and debug.

 Namespace(dataset_path='./data', device='cuda', eval_before_start=False, fp16='O1', gradient_accumulation_steps=8, 
 lm_coef=1.0, local_rank=-1, lr=6.25e-05, max_history=4, max_norm=1.0, max_seq_len=None, mc_coef=1.0, 
 model_checkpoint='gpt2', n_epochs=3, num_candidates=2, personality_permutations=1, subreddit=[], 
 train_batch_size=2, valid_batch_size=2)
 
 get errors:
 /pytorch/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = float, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [123,0,0], thread: [64,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
...
2019-07-14 18:39:04 ASUS-WS ignite.engine.engine.Engine[17369] ERROR Current run is terminating due to exception: CUDA error: device-side assert triggered.
Traceback (most recent call last):
  File "train.py", line 538, in <module>
    train(args)
  File "train.py", line 518, in train
    trainer.run(train_loader, max_epochs=args.n_epochs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 388, in run
    self._handle_exception(e)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 352, in _handle_exception
    raise e
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 375, in run
    hours, mins, secs = self._run_once_on_dataset()
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 341, in _run_once_on_dataset
    self._handle_exception(e)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 352, in _handle_exception
    raise e
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 333, in _run_once_on_dataset
    self.state.output = self._process_function(self, batch)
  File "train.py", line 378, in update
    lm_loss, mc_loss = model(*batch)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 696, in forward
    hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 556, in forward
    hidden_states, present = block(hidden_states, layer_past)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 280, in forward
    a, present = self.attn(self.ln_1(x), layer_past=layer_past)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 240, in forward
    x = self.c_attn(x)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 197, in forward
    x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/apex/amp/wrap.py", line 21, in wrapper
    args[i] = utils.cached_cast(cast_fn, args[i], handle.cache)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/apex/amp/utils.py", line 117, in cached_cast
    casted_x = cast_fn(x)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/apex/amp/utils.py", line 60, in maybe_half
    return x.half()
RuntimeError: CUDA error: device-side assert triggered

--------

train with opt_level ie fp16='O0' (fp32):

/pytorch/aten/src/THC/THCTensorIndex.cu:362: void indexSelectLargeIndex(TensorInfo<T, IndexType>, TensorInfo<T, IndexType>, TensorInfo<long, IndexType>, int, int, IndexType, IndexType, long) [with T = float, IndexType = unsigned int, DstDim = 2, SrcDim = 2, IdxDim = -2, IndexIsMajor = true]: block: [78,0,0], thread: [95,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
2019-07-14 18:51:42 ASUS-WS ignite.engine.engine.Engine[19885] ERROR Current run is terminating due to exception: CUDA error: device-side assert triggered.

Traceback (most recent call last):
  File "train.py", line 538, in <module>
    train(args)
  File "train.py", line 518, in train
    trainer.run(train_loader, max_epochs=args.n_epochs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 388, in run
    self._handle_exception(e)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 352, in _handle_exception
    raise e
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 375, in run
    hours, mins, secs = self._run_once_on_dataset()
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 341, in _run_once_on_dataset
    self._handle_exception(e)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 352, in _handle_exception
    raise e
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/ignite/engine/engine.py", line 333, in _run_once_on_dataset
    self.state.output = self._process_function(self, batch)
  File "train.py", line 378, in update
    lm_loss, mc_loss = model(*batch)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/apex/amp/_initialize.py", line 204, in new_fwd
    **applier(kwargs, input_caster))
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 696, in forward
    hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 556, in forward
    hidden_states, present = block(hidden_states, layer_past)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 280, in forward
    a, present = self.attn(self.ln_1(x), layer_past=layer_past)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 240, in forward
    x = self.c_attn(x)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/walle/.virtualenvs/bert_nlp/lib/python3.7/site-packages/pytorch_pretrained_bert/modeling_gpt2.py", line 197, in forward
    x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
RuntimeError: CUDA error: device-side assert triggered





