import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LlamaForCausalLM


class ConcurrentPredictionOptimizer:
    def __init__(self, model, learning_rate=1e-4, ema_alpha=0.9, num_target_layers=1, leap_steps=20):
        self.model = model
        self.transformer_model = model.model

        self.num_target_layers = num_target_layers
        self.base_layers = torch.nn.ModuleList(self.transformer_model.layers[28-num_target_layers:])
        self.leap_steps = leap_steps
        

        self.embedding = self.transformer_model.embed_tokens
        self.rotary_emb = self.transformer_model.rotary_emb
        self.final_norm = self.transformer_model.norm
        self.lm_head = model.lm_head
        
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha
        
        self.session_start_weights = {}
        self.session_delta_weights = {}
        
        for name, param in self.base_layers.named_parameters():
            self.session_start_weights[name] = param.clone().detach()
            self.session_start_weights[name].requires_grad_(False)

            
            self.session_delta_weights[name] = torch.zeros_like(param)
        
        self.optimizer = optim.SGD(self.base_layers.parameters(), lr=learning_rate, momentum=0)

    def start_session(self, load_delta_path=None):
        with torch.no_grad():
            for name, param in self.base_layers.named_parameters():
                if load_delta_path:
                    loaded_deltas = torch.load(load_delta_path)
                    if name in loaded_deltas:
                        param += loaded_deltas[name]
                        print(f"Loaded and applied delta for {name} from {load_delta_path}")
                    else:
                        print(f"No delta found for {name} in {load_delta_path}, skipping.")
                self.session_start_weights[name] = param.clone().detach()
                self.session_delta_weights[name].zero_()

    def update_session_delta(self, save_path=None):
        """Update cumulative session delta."""
        with torch.no_grad():
            for name, param in self.base_layers.named_parameters():
                current_delta = param.data - self.session_start_weights[name]
                self.session_delta_weights[name] = current_delta
        
        if save_path:
            torch.save(self.session_delta_weights, save_path)
            print(f"Session delta weights saved to {save_path}")
                
    def compute_alignment_loss(self, prediction_logits, actual_tokens, generator_steps):
        
        loss_fct = nn.CrossEntropyLoss()
    
    
        prompt_length = actual_tokens.shape[1] - generator_steps  
    
        if generator_steps % self.leap_steps == 0 and generator_steps > 0:
           train_length = prompt_length + self.leap_steps - 1
        else:
            train_length = prompt_length - 1
    
        shift_logits = prediction_logits[..., :train_length, :].contiguous()
        shift_labels = actual_tokens[..., 1:train_length+1].contiguous()
    
        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), 
                    shift_labels.view(-1))
        return loss

    def adaptive_generation_step(self, input_ids, attention_mask, generator_steps=0):
        """Performs adaptation with session tracking."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.base_layers.parameters():
            param.requires_grad = True
        
        with torch.no_grad():
            outputs = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        static_prediction = outputs.hidden_states[28-self.num_target_layers]
        batch_size, sequence_length = input_ids.shape
        position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(self.embedding(input_ids), position_ids)
        
        hidden_states = static_prediction
        for b_layer in self.base_layers:  
            layer_outputs = b_layer(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)
            hidden_states = layer_outputs[0]
        
        hidden_states = self.final_norm(hidden_states)
        prediction_logits = self.lm_head(hidden_states)
        
        alignment_loss = self.compute_alignment_loss(prediction_logits, input_ids, generator_steps)      
        
        self.optimizer.zero_grad()
        alignment_loss.backward()
        self.optimizer.step()
        
        if generator_steps % self.leap_steps == 0:
            self.update_session_delta()
            print(f"Updated session delta at step {generator_steps}")
        
        return alignment_loss, prediction_logits

    def generate(self, input_ids, max_new_tokens=50):
        """Generation with session management."""
        self.model.eval()
        self.start_session() 
        
        generated_ids = input_ids.clone()
        attention_mask = torch.ones_like(generated_ids)
        
        for step in range(max_new_tokens):
            loss, logits = self.adaptive_generation_step(generated_ids, attention_mask, step)
            
            print(f"Step {step + 1}, Loss: {loss.item():.4f}")
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            if hasattr(self.model.config, 'eos_token_id') and next_token.item() == self.model.config.eos_token_id:
                break
        
        self.update_session_delta()

        print("Session completed - final delta calculated")
        
        return generated_ids

    def apply_session_delta_permanently(self):
        """Apply accumulated session delta to original weights."""
        with torch.no_grad():
            for name, param in self.base_layers.named_parameters():
                self.original_weights[name] += self.session_delta_weights[name]

    def reset_to_session_start(self):
        """Reset to session start weights."""
        with torch.no_grad():
            for name, param in self.base_layers.named_parameters():
                param.data.copy_(self.session_start_weights[name])
