require 'torch'
require 'nn'
require 'image'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'


--[[
Run a trained DenseCap model on images.

The inputs can be any one of:
- a single image: use the flag '-input_image' to give path
- a directory with images: use flag '-input_dir' to give dir path
- MSCOCO split: use flag '-input_split' to identify the split (train|val|test)

The output can be controlled with:
- max_images: maximum number of images to process. Set to -1 to process all
- output_dir: use this flag to identify directory to write outputs to
- output_vis: set to 1 to output images/json to the vis directory for nice viewing in JS/HTML
--]]


local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint',
  'data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-image_size', 720)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-num_proposals', 1000)

-- Input settings
cmd:option('-input_dir', '')
cmd:option('-input_qa_json_file_path', '')

-- Output settings
cmd:option('-max_images', 100, 'max number of images to process')
cmd:option('-output_dir', '')

-- Misc
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
local opt = cmd:parse(arg)


function run_image(model, img_path, opt, dtype)

  -- Load, resize, and preprocess image
  local img = image.load(img_path, 3)
  img = image.scale(img, opt.image_size):float()
  local H, W = img:size(2), img:size(3)
  local img_caffe = img:view(1, 3, H, W)
  img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
  vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
  img_caffe:add(-1, vgg_mean)

  -- Run the model forward
  local boxes, scores, captions = model:forward_test(img_caffe:type(dtype))

  return captions
end

function get_input_images(opt)
  -- utility function that figures out which images we should process 
  -- and fetches all the raw image paths
  local image_paths = {}
  local original_paths = {}

  qa_combined_action_reason = utils.read_json(opt.input_qa_json_file_path)

  for img_path, value in pairs(qa_combined_action_reason)
    do
      local img_in_path = paths.concat(opt.input_dir, img_path)
      table.insert(image_paths, img_in_path)
      table.insert(original_paths, img_path)
  end

  return image_paths, original_paths
end

-- Load the model, and cast to the right type
local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:convert(dtype, use_cudnn)
model:setTestArgs{
  rpn_nms_thresh = opt.rpn_nms_thresh,
  final_nms_thresh = opt.final_nms_thresh,
  num_proposals = opt.num_proposals,
}
model:evaluate()

-- get paths to all images we should be evaluating
local image_paths, original_paths = get_input_images(opt)
local num_process = math.min(#image_paths, opt.max_images)
local results_json = {}

for k=1,num_process do
  local img_path = image_paths[k]
  local original_path = original_paths[k]
  print(string.format('%d/%d processing image %s', k, num_process, img_path))

  -- run the model on the image and obtain results
  local caption = run_image(model, img_path, opt, dtype)  

  results_json[original_path] = caption
end

-- serialize to json
utils.write_json(paths.concat(opt.output_dir, 'results.json'), results_json)
