from model.XVNLI.nli import NLI

fine_tune_nli = NLI(20, "ru")
fine_tune_nli.train()
fine_tune_nli.evaluate()
