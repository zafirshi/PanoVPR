import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def add_mask(train_set, img, img_path_idx, win_len=448):
    pos_mask = np.zeros(train_set.database_resize, dtype=np.float64) + (100 / 255)
    pos_w = img.shape[1]  # 448, 448*8, 3
    mask_left = (pos_w / train_set.split_nums) * img_path_idx
    mask_right = mask_left + win_len

    if mask_left < mask_right <= pos_w:
        pos_mask[:, int(mask_left):int(mask_right)] = 0
    elif mask_left< pos_w < mask_right:
        pos_mask[:int(mask_right-pos_w), int(mask_left):] = 0
    else:
        raise Exception('Adding mask on img goes WRONG!')
    pos_mask_3d = np.stack((pos_mask, pos_mask, pos_mask), axis=2)

    normal_img = (img / 255.).astype(np.float64)
    img = np.clip(normal_img - pos_mask_3d, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


def display_mining(train_set, triplets_global_indexes_array, save_path,  pos_patch_loc, neg_patch_loc):
    full_images_path = np.array(train_set.images_paths)
    db_num = train_set.database_num
    for iter_idx, triple in enumerate(tqdm(triplets_global_indexes_array, ncols=100, desc='Show Mining Results')):
        if iter_idx % 50 == 0:
            query_idx, pos_idx, neg_idx = triple[0], triple[1], triple[2:]
            query_img_path = str(full_images_path[query_idx + db_num])
            pos_img_path = str(full_images_path[pos_idx])
            neg_img_path = full_images_path[neg_idx].tolist()

            query = np.array(path_to_pil_img(query_img_path).resize((448, 448)))
            pos = np.array(path_to_pil_img(pos_img_path).resize((3584, 448)))
            negs = list(map(lambda x: np.array(path_to_pil_img(x).resize((3584, 448))), neg_img_path))
            black = np.zeros((448 * 10, 448, 3), dtype=np.uint8)

            # mask
            pos = add_mask(train_set, pos, pos_patch_loc[iter_idx].item())
            for i, each_neg in enumerate(negs):
                negs[i] = add_mask(train_set, each_neg, neg_patch_loc[iter_idx][i].item())

            right = np.concatenate([pos] + negs, axis=0)
            left = np.concatenate((query, black), axis=0)
            one_in_all = np.concatenate((left, right), axis=1)

            plt.imsave(save_path + f'iter{iter_idx}.png', one_in_all)

            with open(save_path + 'record.txt', 'a+') as f:
                f.write(f'======> iter_idx:{iter_idx}/{train_set.queries_num} <======\n')
                f.write(f'Query_Path: {query_img_path}\n')
                f.write(f'Pos_Path: {pos_img_path}\n')
                for each_neg_path in neg_img_path:
                    f.write(f'Neg_Path: {each_neg_path}\n')


def display_inference(eval_ds, predictions, save_path, focus_patch_loc):
    full_images_path = np.array(eval_ds.images_paths)
    db_num = eval_ds.database_num
    for query_idx, each_query_pred in enumerate(tqdm(predictions, ncols=100, desc='Show Inference Results')):
        if query_idx % 50 == 0:
            query_img_path = str(full_images_path[query_idx + db_num])
            db_img_path = full_images_path[each_query_pred[:5]].tolist()  # 便于展示，选top5

            query = np.array(path_to_pil_img(query_img_path).resize((448, 448)))
            db = list(map(lambda x: np.array(path_to_pil_img(x).resize((3584, 448))), db_img_path))
            black = np.zeros((448 * 4, 448, 3), dtype=np.uint8)

            # add mask
            for i, each_db in enumerate(db):
                db[i] = add_mask(eval_ds, each_db, focus_patch_loc[query_idx][i].item())

            right = np.concatenate(db, axis=0)
            left = np.concatenate((query, black), axis=0)
            one_in_all = np.concatenate((left, right), axis=1)

            # save img
            plt.imsave(save_path + f'query_idx{query_idx}.png', one_in_all)

            with open(save_path + 'inference_list.txt', 'a+') as f:
                f.write(f'======> query_idx:{query_idx}/{eval_ds.queries_num} <======\n')
                f.write(f'Query_Path: {query_img_path}\n')
                for each_db_path in db_img_path:
                    f.write(f'Pred_db_Path: {each_db_path}\n')