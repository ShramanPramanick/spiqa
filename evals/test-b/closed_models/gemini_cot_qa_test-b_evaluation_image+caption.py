# Copyright 2024 Google LLC
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import google.generativeai as genai
import cv2
import random
import json
import os
import argparse
import glob
import time

parser = argparse.ArgumentParser(description='Evaluate on Qasa/Qasper.')
parser.add_argument('--response_root', type=str, help='Response Root path.')
parser.add_argument('--image_resolution', type=int, help='Image Resolution.')
parser.add_argument('--model_id', type=str, help='gpt-4-vision-preview/gpt-4o')
args = parser.parse_args()


genai.configure(api_key="")

qasa_filtered_annotations_path = '../../../datasets/test-B/SPIQA_testB.json'
with open(qasa_filtered_annotations_path, "r") as f:
  qasa_data = json.load(f)

model = genai.GenerativeModel('models/' + args.model_id)


_QASA_IMAGE_ROOT = "../../../datasets/test-B/SPIQA_testB_Images"

def cv2_imagebytes(abs_path):
    image = cv2.imread(abs_path)
    if args.image_resolution == -1:
        resized_image = image
    else:
        resized_image = cv2.resize(image, (args.image_resolution, args.image_resolution))
    image_bytes = cv2.imencode('.png', resized_image)[1].tobytes()
    return image_bytes

def prepare_inputs(paper, question_idx):
  all_figures = list(paper['all_figures_tables'].keys())
  referred_figures = list(set(paper['referred_figures_tables'][question_idx]))
  all_figures_captions = []
  answer = paper['composition'][question_idx]

  if len(all_figures) > 8:
    referred_figures_number = len(referred_figures)
    other_figures_number = 8 - referred_figures_number
    all_other_figures = list(set(all_figures) - set(referred_figures))
    random.shuffle(all_other_figures)
    all_figures_modified = all_other_figures[:other_figures_number] + referred_figures
    random.shuffle(all_figures_modified)
    referred_figures_indices = [all_figures_modified.index(element) for element in referred_figures]

  else:
    all_figures_modified = all_figures
    random.shuffle(all_figures_modified)
    referred_figures_indices = [all_figures_modified.index(element) for element in referred_figures]

  for figure in all_figures_modified:
    all_figures_captions.append(paper['all_figures_tables'][figure])

  all_figures_bytes = {}
  for idx, figure in enumerate(all_figures_modified):
    all_figures_bytes['figure_{}'.format(idx)] = {
      'mime_type': 'image/png',
      'data': cv2_imagebytes(os.path.join(_QASA_IMAGE_ROOT, figure))
    }

  return answer, all_figures_captions, all_figures_bytes, referred_figures_indices, all_figures_modified, referred_figures


# CoT QA
_PROMPT = "You are given a question, a few input images, and a caption corresponding to each input image. \
First, please determine which image and corresponding caption is most helpful to answer the question, and briefly explain why. \
Next, please generate a direct answer to the question. \
Question: <question>. \
First output which image is helpful in the following format: {'Image': A, 'Rationale': 'Very Brief Explanation on Why Image A is helpful'} where A is the image number. \
Next, answer the question as The answer is : <Your Answer>. \n"


def infer_gemini(qasa_data, model):
  
    _RESPONSE_ROOT = args.response_root
    os.makedirs(_RESPONSE_ROOT, exist_ok=True)
  
    for paper_id, paper in qasa_data.items():
        if os.path.exists(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json')):
            continue
        response_paper = {}

        try:
            for question_idx, question in enumerate(paper['question']):

                answer, all_figures_captions, all_figures_bytes, referred_figures_indices, all_figures_modified, referred_figures = prepare_inputs(paper, question_idx)

                contents = [_PROMPT.replace('<question>', question)]

                for idx, figure_bytes in enumerate(list(all_figures_bytes.keys())):
                    contents.append("Image {}: ".format(idx))
                    contents.append(all_figures_bytes[figure_bytes])
                    contents.append("Caption {}: {}".format(idx, all_figures_captions[idx]))
                    contents.append('\n\n')

                response = model.generate_content(
                    contents=contents
                )
                print(response.text)
                print('-------------------------------')
                time.sleep(5)

                question_key = paper['question_key'][question_idx]
                response_paper.update({question_key: {'question': question, 'referred_figures_indices': referred_figures_indices, 'response': response.text,
                                                        'all_figures_names': all_figures_modified, 'referred_figures_names': referred_figures, 'answer': answer}})

        except:
            continue

        with open(os.path.join(_RESPONSE_ROOT, str(paper_id) + '_response.json'), 'w') as f:
            json.dump(response_paper, f)


if __name__ == '__main__':

    infer_gemini(qasa_data, model)
    print(len(glob.glob(args.response_root + '/*.json')))