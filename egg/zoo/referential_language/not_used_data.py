"""
class BboxResizer:
    def __init__(self, new_size: Sequence[int]):
        self.new_sie = new_size

    def __call__(self, original_size: Sequence[int], bbox: List[Sequence[int]]):
        # it assumes original_size is in the W x H format
        ratios = [
            torch.tensor(s) / torch.tensor(s_orig)
            for s, s_orig in zip(self.new_size, original_size)
        ]

        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = bbox

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        new_coords = torch.Tensor((xmin, ymin, xmax, ymax))
        return new_coords
"""

"""
self.queue.extend(batch[idx:])

while curr_batch_size < max_batch_size:
    try:
        elem = self.queue.popleft()
    except IndexError:
        break

    idx += 1
    if curr_batch_size + int(elem[-1].item()) > max_batch_size:
        missing_elems = max_batch_size - curr_batch_size
        elem = (
            elem[0][missing_elems:],
            elem[1][missing_elems:],
            elem[2][missing_elems:],
            torch.Tensor([missing_elems]),
        )

    sender_input.append(elem[0])
    labels.append(elem[1])
    receiver_input.append(elem[2])
    elem_idx_in_batch = torch.Tensor([idx for _ in range(elem[3])])

    curr_batch_size += elem[3].item()

if curr_batch_size < max_batch_size:
    missing_elems = curr_batch_size - max_batch_size
    img_size, channels = sender_input[0][2:], 3

    sender_input.append(torch.zeros(missing_elems, channels, img_size))
    receiver_input.append(torch.zeros(missing_elems, channels, img_size))

    # -100 is missing_index for xent loss
    labels.append(torch.Tensor([-100 for _ in range(missing_elems)]))
    elem_idx_in_batch = torch.Tensor([-100 for _ in range(missing_elems)])

    curr_batch_size += missing_elems
"""


"""
torch.save(
    (
        img,
        sender_img,
        receiver_img,
        coords,
        self.label_name_to_class_description[label_code],
    ),
    f"/private/home/rdessi/dump_img/img_{index}.jpg",
)
"""
