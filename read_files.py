from inspect_ai.log import read_eval_log

log = read_eval_log("/work/training/MiniMoE/logs_english/baseline_tau2_airline.eval")

# Top-level info
print(log.status)       # "success", "error", etc.
print(log.eval.model)   # model used
print(log.eval.task)    # task name

# Results/scores
print(log.results)

# Individual samples
# for sample in log.samples:
#     print(sample.input)   # the prompt
#     print(sample.output)  # model response
#     print(sample.scores)  # scores for this sample