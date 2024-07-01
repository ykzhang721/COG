import torch
from torch import nn
from typing import Any, Optional
import types
from numpy import *
import json


class MyModel(nn.Module):
    def __init__(self, ptm_model, processor, n_con, device, con_type, id2con, model_name, method):
        super().__init__()
        self.encoder = ptm_model
        self.tokenizer = processor
        self.n_con = n_con
        self.method = method
        self.device = device 
        self.id2con = id2con
        self.model_name = model_name
        contype2trans = {'onlyent': 'trans_only_entity.json', 
                        'blc': 'trans_blc.json', 
                        'all': 'trans_all.json'}
        if self.method != "onlyent":
            contype = contype2trans[con_type]
        else:
            contype = contype2trans['onlyent']
        with open(f'resources/{contype}','r') as f:
            self.trans = json.load(f)


        if model_name == 'align':
            self.encoder.align_forward = types.MethodType(align_forward, self.encoder)
        if model_name == 'clip':
            self.encoder.clip_forward = types.MethodType(clip_forward, self.encoder)
        if model_name == 'blip':
            self.encoder.blip_forward = types.MethodType(blip_forward, self.encoder)

        self.encoder.to(self.device)

    def train_forward(self, imgs, cons, ents):
        sigmoid = torch.nn.Sigmoid()
        encoded_imgs = self.encoder.get_image_features(imgs)
        preds = []
        if ents:
            for i_e, ent in enumerate(ents):
                ent = self.trans[ent]
                inputs = self.tokenizer(text=ent, return_tensors="pt", padding=True)
                inputs['image_embeds'] = encoded_imgs
                inputs = inputs.to(self.device)
                if self.model_name == 'align':
                    outputs = self.encoder.align_forward(**inputs)
                if self.model_name == 'clip':
                    outputs = self.encoder.clip_forward(**inputs)
                if self.model_name == 'blip':
                    outputs = self.encoder.blip_forward(**inputs)
                preds.append(sigmoid(outputs))
        if cons:
            for i_e, ent_cons in enumerate(cons):
                word_cons = [self.id2con[con] for con in ent_cons.tolist()]
                word_cons_en = [self.trans[w] for w in word_cons]
                inputs = self.tokenizer(text=word_cons_en, return_tensors="pt", padding=True)
                inputs['image_embeds'] = encoded_imgs
                inputs = inputs.to(self.device)
                if self.model_name == 'align':
                    outputs = self.encoder.align_forward(**inputs)
                if self.model_name == 'clip':
                    outputs = self.encoder.clip_forward(**inputs)
                if self.model_name == 'blip':
                    outputs = self.encoder.blip_forward(**inputs)
                preds.append(sigmoid(outputs))
        return preds


    def eval_forward_tc(self, imgs, cons, ents, cfhe, hyperparams, save_evidence=False):
        sigmoid = torch.nn.Sigmoid()
        preds = []
        evidence = {}
        for i_e, ent in enumerate(ents):
            encoded_imgs = self.encoder.get_image_features(imgs[i_e].unsqueeze(0))
            ent_en = self.trans[ent]
            inputs = self.tokenizer(text=ent_en, return_tensors="pt", padding=True)
            inputs['image_embeds'] = encoded_imgs
            inputs.to(self.device)
            if self.model_name == 'align':
                outputs = self.encoder.align_forward(**inputs)
            if self.model_name == 'clip':
                outputs = self.encoder.clip_forward(**inputs)
            if self.model_name == 'blip':
                outputs = self.encoder.blip_forward(**inputs)
            p = sigmoid(outputs.squeeze(1))
            pred = (p >= hyperparams['threshold']).int()
            
            if cons and p.item() <= hyperparams['threshold']:
                word_cons = [self.id2con[con] for con in cons[i_e].tolist()]
                word_cons_en = [self.trans[w] for w in word_cons]
                inputs = self.tokenizer(text=word_cons_en, return_tensors="pt", padding=True)
                inputs['image_embeds'] = encoded_imgs
                inputs.to(self.device)
                if self.model_name == 'align':
                    outputs = self.encoder.align_forward(**inputs)
                if self.model_name == 'clip':
                    outputs = self.encoder.clip_forward(**inputs)
                if self.model_name == 'blip':
                    outputs = self.encoder.blip_forward(**inputs)

                outputs = sigmoid(outputs)
                a = outputs.squeeze(0)
                
                a = list(map(float,a))
                cf_h_e_ent = cfhe[ent]
                aa = [a[i_e] * cf_h_e_ent[con] for i_e , con in enumerate(word_cons)]
                p = torch.tensor(sum(aa)/len(aa), dtype=torch.double).unsqueeze(0)
                pred = (p >= hyperparams['threshold']).int()
                evidence[ent_en] = {self.trans[con]:{"probability":round(a[i],2),"contribution":round(cf_h_e_ent[con],2)} for i, con in enumerate(word_cons)}

            pred = pred.to(self.device)
            preds.append(pred)
        
        preds = torch.cat(preds, dim=0)
        if save_evidence:
            return preds, evidence
        else:
            return preds

    def eval_forward_lp(self, imgs, ents):
        sigmoid = torch.nn.Sigmoid()
        preds = []
        for i_e, ent in enumerate(ents):
            encoded_imgs = self.encoder.get_image_features(imgs[i_e])
            ent_en = self.trans[ent]
            inputs = self.tokenizer(text=ent_en, return_tensors="pt", padding=True)
            all_output = []
            for i in encoded_imgs:
                inputs['image_embeds'] = i
                inputs.to(self.device)
                if self.model_name == 'align':
                    outputs = self.encoder.align_forward(**inputs)
                if self.model_name == 'clip':
                    outputs = self.encoder.clip_forward(**inputs)
                if self.model_name == 'blip':
                    outputs = self.encoder.blip_forward(**inputs)
                p = sigmoid(outputs)
                all_output.append(p)
            pred = torch.cat(all_output, dim=0)
            pred = pred.to(self.device)
            preds.append(pred)
        return preds



def align_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # pixel_values: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # vision_outputs = self.vision_model(
        #     pixel_values=pixel_values,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # image_embeds = vision_outputs[1]
        text_embeds = text_outputs[0][:, 0, :]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) / self.temperature
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = align_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        # return AlignOutput(
        #     loss=loss,
        #     logits_per_image=logits_per_image,
        #     logits_per_text=logits_per_text,
        #     text_embeds=text_embeds,
        #     image_embeds=image_embeds,
        #     text_model_output=text_outputs,
        #     vision_model_output=vision_outputs,
        # )
        return logits_per_image


def clip_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        #pixel_values: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        '''
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        '''

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return logits_per_image 



def blip_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # pixel_values: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Use BLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # vision_outputs = self.vision_model(
        #     pixel_values=pixel_values,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # image_embeds = vision_outputs[1]
        # image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = blip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return logits_per_image