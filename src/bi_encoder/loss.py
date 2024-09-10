import torch
from torch.nn import functional as F

def dot_product_scores(q_vectors, doc_vectors):
    r = torch.matmul(q_vectors, doc_vectors.permute(1, 0))
    return r
def cosine_scores(q_vectors, doc_vectors):
    r = F.cosine_similarity(q_vectors, doc_vectors, dim=0)
    return r

class BiEncoderNllLoss(object):
    """
        Negative log likelihood loss:
    """
    def __init__(self,
                 score_type="dot"):
        self.score_type = score_type
    def calc(self,
             q_vectors,
             doc_vectors
            ):
        scores = self.get_scores(q_vectors, doc_vectors)
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size()[0]
            scores = scores.view(q_num, -1)
            positive_idx_per_ques = [i for i in range(q_num)]
        
        softmax_scores = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_ques).to(softmax_scores.device),
            reduction='mean'
        )
        max_scores, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_ques).to(max_idxs.device)).sum()
        return loss, correct_predictions_count
        
    def get_scores(self, q_vectors, doc_vectors):
        if self.score_type == 'dot':
            return dot_product_scores(q_vectors, doc_vectors)
        elif self.score_type == 'cosine':
            return cosine_scores(q_vectors, doc_vectors)
            