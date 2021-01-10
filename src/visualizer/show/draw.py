from PIL import Image
import cv2


def load_image(path, scale=1.0):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
        image = np.asarray(image) * scale / 255.0
        return image


def draw_ped_ann(image, bbox, action):
    thickness = 2
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    if action:
        color = (0, 0, 255)
        org = (int(bbox[0] - 10), int(bbox[1] - 10))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .75
        image = cv2.putText(image, 'walking', org, font, fontScale, color, 2, cv2.LINE_AA)
    else:
        color = (0, 255, 0)
        org = (int(bbox[0]), int(bbox[1] - 10))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        image = cv2.putText(image, 'standing', org, font, fontScale, color, 2, cv2.LINE_AA)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image


def show_frame_in_history(sample, k, title=''):
    bbox = sample['bbox'][k]
    action = sample['action'][k]
    s = 'walking' if action else 'standing'
    img_tensor = sample['image'][k]
    pid = sample['id']
    Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
    pil_img = Tensor2PIL(img_tensor)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image = draw_ped_ann(cv2_img, bbox, action)
    plt.title(f'{pid}, image {k} ({s})')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),aspect='equal')
    
    return None