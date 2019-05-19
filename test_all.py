import baseline
import test_multitask
import lstm


def test_all():
    baseline.main()
    lstm.lstm_memory_efficient()
    test_multitask.multitask_memory_efficient()
    test_multitask.multitask_smhd()
    

if __name__ == "__main__":
    test_all()