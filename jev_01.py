import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


MODEL_PATH = "./qwen2.5-0.5b"


class JevV1(nn.Module):

    def __init__(self, model_path, number_of_choices):
        super().__init__()

        # -----------------------------------
        # 1. Load ONLY the Qwen backbone
        # -----------------------------------

        self.backbone = AutoModel.from_pretrained(
            model_path,
            torch_dtype="auto"
        )

        hidden_size = self.backbone.config.hidden_size

        print("Qwen hidden size:", hidden_size)

        # -----------------------------------
        # 2. OUR architecture starts here
        # -----------------------------------

        self.decision_head = nn.Sequential(

            nn.Linear(hidden_size, 256),

            nn.SiLU(),

            nn.Linear(
                256,
                number_of_choices
            )
        )


    def forward(
        self,
        input_ids,
        attention_mask
    ):

        # -----------------------------------
        # Qwen understands the input
        # -----------------------------------

        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )

        # Shape:
        #
        # [batch, tokens, 896]

        hidden = output.last_hidden_state


        # -----------------------------------
        # Take representation of last token
        # -----------------------------------

        last_token_index = (
            attention_mask.sum(dim=1) - 1
        )

        batch_index = torch.arange(
            hidden.size(0),
            device=hidden.device
        )

        decision_vector = hidden[
            batch_index,
            last_token_index
        ]


        # -----------------------------------
        # OUR decision head
        # -----------------------------------

        logits = self.decision_head(
            decision_vector.float()
        )


        # Direct probabilities
        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        return probabilities


# =========================================
# TEST
# =========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

choices = [
    "refund",
    "cancel",
    "technical"
]

model = JevV1(
    MODEL_PATH,
    number_of_choices=len(choices)
)

text = """
STATE:
The customer says:
Cancel my subscription immediately.

QUESTION:
What is the customer's intent?
"""

inputs = tokenizer(
    text,
    return_tensors="pt"
)

with torch.no_grad():

    probabilities = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )


print("\nRESULT\n")

for choice, probability in zip(
    choices,
    probabilities[0]
):

    print(
        choice,
        round(probability.item(), 4)
    )