import argparse
from collections import Counter
from decoder_utils import *
import time
from grader import math_equal

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args.model_path, args.classifier_path, args.insert_layer)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    total = 0
    correct_total = 0
    with open(args.log_dir, "w") as wp:

        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1:
                continue
            output_line = {}
            
            print("*"*25)
            print("{}st data".format(i+1))
            start_time = time.time()
                    
            # Prepare question template ...
            x, y = data
            x = "Solve this math problem step by step. You MUST put your final answer in \\boxed{}. Problem: " + x[0] + "\n Solution:\n<think>\n"
            #x = f"Q: {x[0]}\nA: "
            y = y[0].strip()
            
            # print(x, y)
            
            output_line["question"] = x
            output_line["gold_ans"] = y

            preds = []
            # iterations for self-consistency, default is 1
            for _ in range(args.iterations):            
                # Answer experiment by generating text ...
                z = decoder.decode(x, args.model_path, method=args.method, max_length=args.max_length, temperature=args.temperature)
                output_line["rationale"] = z

                # Answer extraction for zero-shot-cot ...
                if args.answer_extract == "zero_shot_cot":
                    z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                    pred = decoder.decode(z2, args.model_path, method="regular", max_length=30)
                    print(z2 + pred)
                else:
                    pred = z
                    print(x + pred)

                # Clensing of predicted answer ...
                pred = answer_cleansing(args, pred)
                if pred:
                    preds.append(pred)
            
            if not preds:
                preds.append('')
            pred = Counter(preds).most_common(1)[0][0] 
            output_line["pred_ans"] = pred
            output_line["wrap_que"] = x

            output_json = json.dumps(output_line)
            wp.write(output_json + '\n')
            end_time = time.time()

            correct = int(math_equal(pred, y)) #(np.array([pred]) == np.array([y])).sum().item()

            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print("Assert : " + str(bool(correct)))
            print("Time cost: {:.2f}".format(end_time - start_time))
            print("*"*25)
            
            # Checking answer ...
            correct_total += correct
            total += 1 #np.array([y]).size(0)
            accuracy = (correct_total * 1.0 / total) * 100
            print("current accuracy : {:.2f}".format(accuracy))
            
            if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                break
                #raise ValueError("Stop !!")

    # Calculate accuracy ...
    accuracy = (correct_total * 1.0 / total) * 100
    print("final accuracy : {:.2f}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Reasoning drift steering")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters", "object_tracking", "bigbench_date", "gsmic", "aime2024", "aime2025", "math500", "gpqa"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")

    parser.add_argument("--answer_extract", type=str, default="zero_shot", choices=["zero_shot", "zero_shot_cot"], help="method for answer extraction")
    
    parser.add_argument(
        "--log_dir", type=str, default="./log", help="log directory"
    )
    parser.add_argument(
        "--max_length", type=int, default=4096, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="temperature for LLMs"
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="self consistency iterations"
    )
    parser.add_argument(
        "--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="LLMs model path"
    )
    parser.add_argument(
        "--classifier_path", type=str, default="../distill_bert/model", help="behavior classifier path"
    )
    parser.add_argument(
        "--method", type=str, default="steering", choices=["regular", "beam", "dola", "steering"], help="method"
    )
    parser.add_argument(
        "--insert_layer", type=int, default=23, help="insert layer for steering vectors"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer choice is ("
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "gsmic":
        args.dataset_path = "./dataset/gsmic/GSM-IC_2step.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "gpqa":
        args.dataset_path = "./dataset/GPQA/gpqa_main.csv"
        args.direct_answer_trigger = "\nTherefore, among A through D, the answer choice is ("
    elif args.dataset == "aime2024":
        args.dataset_path = "./dataset/aime/aime_2024_problems.parquet"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "aime2025":
        args.dataset_path = "./dataset/aime/aime2025.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "math500":
        args.dataset_path = "./dataset/math500/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer is "
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is "
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is "
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer choice is ("
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer choice is ("
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the (Yes or No) answer is: "
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is "
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    
    return args

if __name__ == "__main__":
    main()
