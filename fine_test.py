from model.XVNLI.fine_tune_nli import FINE_TUNE_NLI

fine_tune_nli = FINE_TUNE_NLI(20, "ru")
fine_tune_nli.train()
fine_tune_nli.evaluate()
