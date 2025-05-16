import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    args = parser.parse_args()
    
    gt = np.load(args.gt)
    pred = np.load(args.pred)
    
    labels = gt["labels"]
    base_addr = gt["base_addr"]
    masks = gt["mask"]
    pred_labels = pred["pred"]
    
    # find the indices where labels and pred_labels are different
    diff_indices = np.where((labels != pred_labels) & masks)[0] + base_addr
    
    # print the diff_indices
    print(diff_indices)
    
if __name__ == "__main__":
    main()
    
    
    