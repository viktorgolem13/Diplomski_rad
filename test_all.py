import baseline
import test_multitask
import test_multitask2
import lstm


def test_all():
    f = open("results.txt", "w")
    # test_acc, f1_score = baseline.main(train_set_size=100000)
    # f.write("acc: " + str(test_acc))
    # f.write(" f1: " + str(f1_score))
    # f.write("\n")

    acc1, f1 = lstm.lstm_memory_efficient_simple(reload_data=True, data_per_iteration=99, num_of_load_iterations=259)
    f.write("acc: " + str(acc1))
    f.write(" f1: " + str(f1))
    f.write("\n")
    #
    # acc1, f11, acc2, f12 = test_multitask.multitask_smhd_memory_efficient(reload_data=True, data_per_iteration=10,
    #                                                                       num_of_load_iterations=100)
    # f.write("acc1: " + str(acc1))
    # f.write(" f11: " + str(f11))
    # f.write("acc2: " + str(acc2))
    # f.write(" f12: " + str(f12))
    # f.write("\n")

    # acc1, f11, acc2, f12 = test_multitask.multitask_memory_efficient()
    # f.write("acc1: " + str(acc1))
    # f.write(" f11: " + str(f11))
    # f.write("acc2: " + str(acc2))
    # f.write(" f12: " + str(f12))
    # f.write("\n")
    # test_multitask2.multitask_smhd_memory_efficient(reload_data=True, data_per_iteration=5, num_of_load_iterations=100)
    f.close()


if __name__ == "__main__":
    test_all()
