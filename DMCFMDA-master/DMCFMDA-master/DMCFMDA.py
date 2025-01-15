import torch as th
from torch import nn,einsum
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
from get_microbe import GraphTransformer as GraphTransformer1
from get_disease import GraphTransformer as GraphTransformer2
from GAT import GAT



class DMCFMDA(nn.Module):
    def __init__(self, args):
        super(DMCFMDA, self).__init__()
        self.args = args
        self.lin_m = nn.Linear(args.microbe, args.in_feats, bias=False)
        self.lin_d = nn.Linear(args.disease_number, args.in_feats, bias=False)

        self.gt_microbe = GraphTransformer1(device, args.gt_layer, args.microbe, args.microbe,
                                            args.microbe, args.gt_head, args.dropout)
        self.gt_disease = GraphTransformer2(device, args.gt_layer, args.disease_number, args.disease_number,
                                            args.disease_number, args.gt_head, args.dropout)

        self.trans_gat_microbe = GAT(args.microbe, [128, 64], args.out_feats, num_heads=10, dropout=args.dropout)
        self.trans_gat_disease = GAT(args.disease_number, [128, 64], args.out_feats, num_heads=10,
                                          dropout=args.dropout)
        self.gat_md = GAT(args.in_feats, [128, 64], args.out_feats, num_heads=10, dropout=args.dropout)

        self.mlp = nn.Sequential()
        self.dropout = nn.Dropout(args.dropout)
        in_feat = 2 * args.out_feats
        for idx, out_feat in enumerate(args.mlp):
            if idx == 0:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('elu', nn.ELU())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat
            else:
                self.mlp.add_module(str(idx), nn.Linear(in_feat, out_feat))
                self.mlp.add_module('sigmoid', nn.Sigmoid())
                self.mlp.add_module('dropout', nn.Dropout(p=0.2))
                in_feat = out_feat

    def forward(self, mm_graph, dd_graph, md_graph, microe, disease, samples, epoch, Issne=False):
        mm_graph = mm_graph.to(device)
        dd_graph = dd_graph.to(device)
        md_graph = md_graph.to(device)
        microe = microe.to(device)
        disease = disease.to(device)

        mi_sim = self.gt_microbe(mm_graph, microe)    #gt
        di_sim = self.gt_disease(dd_graph, disease)   #gt

        emb_m = self.trans_gat_microbe(mm_graph, mi_sim, microe)
        emb_d = self.trans_gat_disease(dd_graph, di_sim, disease)

        combined_features = th.cat((self.lin_m(microe), self.lin_d(disease)), dim=0)
        emb_md = self.gat_md(md_graph, combined_features, combined_features)

        emb_mm_ass = emb_md[:self.args.microbe, :]
        emb_dd_ass = emb_md[self.args.microbe:, :]

        emb_mm = emb_m + emb_mm_ass
        emb_dd = emb_d + emb_dd_ass

        emb = th.cat((emb_mm[samples[:, 0]], emb_dd[samples[:, 1]]), dim=1)
        result = self.mlp(emb)
        return result, emb_m, emb_mm_ass, emb_d, emb_dd_ass

