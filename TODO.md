# TODO

- Add an optional shared-weights teacher wrapper for distillation so the teacher can reuse the student's frozen backbone tensors in memory. The tricky part is keeping trainable added-token embedding/LM-head rows isolated so the teacher remains a true frozen reference even when the student updates those rows.
