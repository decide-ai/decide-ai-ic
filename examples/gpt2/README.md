# `gpt2`

ic-file-uploader model append_safetensors_bytes ~/.cache/huggingface/hub/models--vicgalle--gpt2-open-instruct-v1/snapshots/eccf4d5899c24523625fe3d41f1cf78c755821b0/model.safetensors
dfx canister call model store_safetensors_bytes_to_stable
dfx canister call model load_safetensors_bytes_from_stable

ic-file-uploader model append_config_bytes ~/.cache/huggingface/hub/models--vicgalle--gpt2-open-instruct-v1/snapshots/eccf4d5899c24523625fe3d41f1cf78c755821b0/config.json
dfx canister call model setup_model

input token sequence, generate token length, sampling temperature

dfx canister call model inference '(vec {1; 2}, 1:nat8, 0.2:float64)'

ic-file-uploader model append_tokenizer_bytes ~/.cache/huggingface/hub/models--vicgalle--gpt2-open-instruct-v1/snapshots/eccf4d5899c24523625fe3d41f1cf78c755821b0/tokenizer.json
dfx canister call model store_tokenizer_bytes_to_stable
dfx canister call model load_tokenizer_bytes_from_stable
dfx canister call model setup_tokenizer

dfx canister call model tokenize 'what is the capital of France?'

text, generate token length, sampling temperature

dfx canister call model generate '("what is the capital of France?", 10:nat8, 0.2:float64)'

the maximum length of token generations is dependent on the input length


export const EOT = 50257;
export const PREFIX_TOKENS = [50258, 198];
export const SUFFIX_TOKENS = [628, 50259, 198];

npm install @vitejs/plugin-react --save-dev

npm install class-variance-authority clsx tailwind-merge lucide-react
npm install @radix-ui/react-slider @radix-ui/react-slot



npm install -g @shadcn/ui
shadcn-ui init

npm install -D @shadcn/ui
npx shadcn-ui init


npx shadcn-ui@latest add slider
npx shadcn-ui@latest add button
npx shadcn-ui@latest add input
npx shadcn-ui@latest add card


