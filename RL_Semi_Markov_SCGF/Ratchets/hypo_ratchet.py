import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Exponential, Gamma, MixtureSameFamily, Dirichlet, Normal
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy as sp

hidden=3
batch_size=16
T=30000
policy_clip = 0.2
if  torch.cuda.is_available():

    device = torch.device("cuda")
    print("here",torch.cuda.get_device_name(0))
    
else:
     device=torch.device("cpu")
########################NN Models############################

class Actor(nn.Module):
      
      def __init__(self):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(3,64)
        self.affine2= nn.Linear(64,32)
        self.affine3 = nn.Linear(32,16)  
        self.action_head_mov = nn.Linear(16,2)
        self.log_probs=[]

      def forward(self, x):
        #print(x)s
       
        x1 = F.tanh(self.affine1(x))          
        x2 = F.tanh(self.affine2(x1))  
        x2 = F.tanh(self.affine3(x2))    
        move_params= (self.action_head_mov(x2))
       
        return F.softmax(move_params,dim=1)
        #F.softmax(weights,dim=1)

class Critic(nn.Module):
      
      def __init__(self):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(3,64)
        self.affine2=nn.Linear(64,32)
        #self.affine3=nn.Linear(32,32)
        self.critic_head = nn.Linear(32, 1)
        self.state_value=[]
      
      
      def forward(self, x):
        x = F.tanh(self.affine1(x))
        x=F.tanh(self.affine2(x))
        #x=F.tanh(self.affine3(x))  
        state_value =self.critic_head(x)
        return state_value


class tau_Net(nn.Module):
      
      def __init__(self,hidden=4):
        super(tau_Net, self).__init__()
        self.hidden=hidden
        self.affine1 = nn.Linear(2,32)     
        self.action_head1 = nn.Linear(32, self.hidden)
        self.action_head2= nn.Linear(32, self.hidden)
        self.action_head3 = nn.Linear(32, self.hidden)
    
        self.log_probs=[]

      def forward(self, x):
        #print(x)s
       
        x1 = F.tanh(self.affine1(x))          
        rates=self.action_head1(x1)
        concs = self.action_head2(x1)
        weights = self.action_head3(x1)   
        
       
        return rates,concs, F.softmax(weights,dim=1)
       
        

def weights_init_uniform(m):
    classname=m.__class__.__name__
    if classname.find('Linear')!=-1:
          torch.nn.init.xavier_uniform_(m.weight)
          m.bias.data.fill_(0.0001)
 
#def experience_replay_buffer():

def count(model):       
       total_params = 0
       for name, parameter in model.named_parameters():
           if not parameter.requires_grad:
              continue
           params = parameter.numel()
           #table.add_row([name, params])
           total_params += params
       print(total_params)
       print(f"Total Trainable Params: {total_params}")
       return total_params     


       
      



policy_learning_rate=1e-4
value_learning_rate=1e-4
actor_model=Actor()
tau_model = tau_Net(hidden)

critic_model=Critic()
actor_model.to(device)
critic_model.to(device)

#copy_actor = Actor(hidden)
target_net= Critic()
#copy_actor.to(device)
target_net.to(device)
actor_model.apply(weights_init_uniform)
critic_model.apply(weights_init_uniform)
target_net.apply(weights_init_uniform)

for p in target_net.parameters():
            p.requires_grad = False

	
def reset_nets():
	actor_model.apply(weights_init_uniform)
	critic_model.apply(weights_init_uniform)
	target_net.apply(weights_init_uniform)

optimizer_actor=optim.Adam(actor_model.parameters(),lr=policy_learning_rate)
optimizer_tau=optim.Adam(tau_model.parameters(),lr=policy_learning_rate)
optimizer_critic=optim.Adam(critic_model.parameters(),lr=value_learning_rate)


num_params_actor1 = count(actor_model)
num_params_actor2 = count(tau_model)
num_params_critic = count(critic_model)
print("approx_num_params", num_params_actor1+num_params_actor2+num_params_critic)

##########################################################################################################################################

        

        


def get_action(x_state,tau_state, bias):
    #print(x_state)
    x=torch.cat((x_state,tau_state),dim=1)
    action_probs = actor_model(x)
    m1=Categorical(action_probs)
    action=m1.sample()
    log_prob_transition=m1.log_prob(action)
    total_log_prob= log_prob_transition.unsqueeze(1)
    return log_prob_transition.unsqueeze(1), action.unsqueeze(1), total_log_prob


def get_tau(x):
    log_rates, log_concs, weights = tau_model(x)
    rates,concs = torch.exp(log_rates),torch.exp(log_concs)
    mix = Categorical(weights)
    comp =Gamma(concs, rates)    
    gmm = MixtureSameFamily(mix, comp)
    waiting_time= gmm.sample()
    log_prob_waiting=gmm.log_prob(waiting_time)
    return waiting_time.unsqueeze(1), log_prob_waiting.unsqueeze(1)

def get_next_q(x_state,tau_state):
    x=torch.cat((x_state,tau_state),dim=1)
    q_value=target_net(x)
    return q_value
    
def get_q_value(x_state,tau_state):
    x=torch.cat((x_state,tau_state),dim=1)   
    q_value=critic_model(x)
    return q_value

biases=np.arange(-2,2.2,0.2)
reward_learning_rate=5e-3
############# Learning Loops#####################################
def main():
    avg=[]    
    avg_current=[]
    for bias in biases:
        
        reset_nets()
        x_state_1 = torch.zeros((batch_size,1),dtype=torch.float).to(device)
        x_state_2 = torch.zeros((batch_size,1),dtype=torch.float).to(device)
        x_state_2[:int(batch_size/2),0] = 1.0 
        x_state = torch.cat((x_state_1,x_state_2),dim=1)
        wait_state=  torch.rand((batch_size,1),dtype=torch.float).to(device) 
        next_state_x = torch.zeros((batch_size,2),dtype=torch.float).to(device)
        t_state = torch.zeros((batch_size,1),dtype=torch.float).to(device)
        current = torch.zeros((batch_size,1),dtype=torch.float).to(device)
        
        TD_error=[]
        total_mean_scgf=[]
        # long time loop while learning
        final_rate=0    
        epochs=1
        average_reward=torch.zeros((batch_size,1),dtype=torch.float).to(device)
        avg_j=0
        t=[]
        time1=time.time()
        for epoch in range(epochs):
            if epoch % 5 == 0:
               print(f"Epoch {epoch}")
            #reset_grads()
            #average_reward = 0
            s=0
            rewards=[]	
            values=[]
            targets=[]
            done=False	
            sim_time=torch.zeros((batch_size,1),dtype=torch.float).to(device)
            i=0
            dist1 = Dist() 
            while not done:
		    # select action from policy
		    #action_rates = select_action(state)
		    # take the action
                if i%1000==0:
                   print(f"-------------------------------------------------sim_time:{sim_time.mean()},episode,{epoch},k,{bias},{average_reward.mean()},{wait_state.mean()}")
                #print(wait_state)
                log_prob_trans, action, total_log_prob=get_action(x_state/2.0, wait_state,bias)
                #print(wait_state, F.sigmoid(wait_state))
                #################################################################################################
              
                
                p2=2
                p1=1
                q=(p1*p2)/(p1+p2)
                r=1
                
                
               
                y = x_state.clone()

                ################################################################
                
                for j in range(batch_size):
                    if action[j,0]==0:   
                        if x_state[j,1]==0:
                           next_state_x[j,0] =  (y[j,0]+1)%(3)
                           next_state_x[j,1] = y[j,1]
                          #print(next_state_x)
                        else:
                           next_state_x[j,0] =  (y[j,0]-1)%(3)
                           next_state_x[j,1] =  y[j,1]
                    else:
                       next_state_x[j,0]=y[j,0] 
                       next_state_x[j,1] = 1-y[j,1]
                
                
                
                
                
                
                
                
                mask = action.clone()
                
                #tau1_f= dist1.rvs(size=batch_size)#
                #tau1_r = np.random.exponential(1/(q+r),size=batch_size)
                #print(tau1_f)
                #tau1_f=torch.tensor(tau1_f,dtype=torch.float).to(device).unsqueeze(1)
                #tau1_r=torch.tensor(tau1_r,dtype=torch.float).to(device).unsqueeze(1)
                #print(tau1_r.shape)
                #tau1=torch.zeros((batch_size,1),dtype=torch.float).to(device)
                #tau1[next_state_x[:,1]==0]=tau1_f[next_state_x[:,1]==0]
                #tau1[next_state_x[:,1]==1]=tau1_r[next_state_x[:,1]==1]
                #print(tau1,tau1_f,tau1_r)
                waiting_time,log_prob_waiting= get_tau(next_state_x)
                tau1 = waiting_time.detach()
                next_state_t= ((t_state)+tau1) 
                
                
                tau=wait_state
                bias_tensor= torch.zeros((batch_size,1)).to(device)
                log_org_waiting=torch.zeros((batch_size,1)).to(device)
                log_org_prob = torch.zeros((batch_size,1)).to(device)
                pdf_FF= ((p1*p2)/(p1-p2))*(torch.exp(-p2*tau)- torch.exp(-p1*tau))
                phi_F = (1/(p2-p1))*((p2*torch.exp(-p1*tau))-(p1*torch.exp(-p2*tau)))
                pdf_FF1= ((p1*p2)/(p1-p2))*(torch.exp(-p2*tau1)- torch.exp(-p1*tau1))
                phi_F1 = (1/(p2-p1))*((p2*torch.exp(-p1*tau1))-(p1*torch.exp(-p2*tau1)))                
                
                #pdf_r = (r)*torch.ones((batch_size,1)).to(device)
                pdf_r1 = (q+r)*torch.ones((batch_size,1)).to(device)
                comp1=pdf_FF
                comp2 = r * phi_F  
                comp11=pdf_FF1
                comp21 = r * phi_F1  
             
                log_org_waiting[next_state_x[:,1]==0]= torch.log(comp11[next_state_x[:,1]==0] + comp21[next_state_x[:,1]==0]) - (r* tau1[next_state_x[:,1]==0])
                log_org_waiting[next_state_x[:,1]==1]= torch.log(pdf_r1[next_state_x[:,1]==1]) -((q+r)*tau1[next_state_x[:,1]==1]) 
                mask[x_state[:,1]==1] = action[x_state[:,1]==1] + 2
                log_org_prob[mask==0]= comp1[mask==0]/(comp1[mask==0] + comp2[mask==0])
                log_org_prob[mask==1]= comp2[mask==1]/(comp1[mask==1] + comp2[mask==1]) 
                log_org_prob[mask==2]=(q/(q+r))
                log_org_prob[mask==3]=(r/(q+r))
                bias_tensor[mask==0]=bias
                bias_tensor[mask==2]=-bias
                I =   log_prob_trans.detach() - torch.log(log_org_prob) + log_prob_waiting.detach() - log_org_waiting 
                #print(I.mean())
                reward= (bias_tensor - I)
                
                sim_time += tau1
                
                t.append(sim_time.mean())
                if sim_time.mean()>=T:
                   done= True
                
                ###########################################################################################################
                
         
                #replay.cache((x_state, t_state, next_state_x,next_state_t, reward, done))
                
                ###########################################################################
                
                
                #x_state,t_state,next_state_x, next_state_t, action, reward, done = replay.recall()
                #print("next",I)
                       
                with torch.no_grad():
                     #next_waiting_time, _, _ ,_ ,_= get_action(next_state_t,next_state_x)
                     next_value=get_next_q(next_state_x/2.0,tau1)
                     #old_log_probs = get_action_logprob(x_state/2.0, F.tanh(0.1*wait_state))
                    
                value=get_q_value(x_state/2.0, tau)
                target=(next_value)+ ((reward) - (tau1  * average_reward)) 
                td_error=(target-value.detach())  
                TD_error.append(td_error.mean())
                policy_loss1 = torch.mean(-(total_log_prob) * (td_error.mean())) 
                policy_loss2 = torch.mean(-(log_prob_waiting) * (td_error.mean())) 
            
                
               
                critic_loss=F.mse_loss(value,target)
                #loss=policy_loss.item()
                optimizer_actor.zero_grad()
                optimizer_tau.zero_grad()
                optimizer_critic.zero_grad()
                
             
                #loss_value=critic_loss.item()   
                policy_loss1.backward()
                policy_loss2.backward()
                critic_loss.backward()     
                optimizer_actor.step()
                optimizer_tau.step()
                optimizer_critic.step()
            
                #scheduler1.step()
                #scheduler2.step()
                
                #L.append(loss)  
                #L_val.append(loss_value)
                #writer.add_scalar(f"Loss/value_loss_{bias}", loss_value,i)
                #writer.add_scalar(f"Loss/policy_loss_{bias}",loss,i)
                #writer.add_scalar(f"Loss/td_error_{bias}",td_error.mean(),i)
               # writer.add_scalar(f"Loss/reward_{bias}",reward.mean(),i)
                
                
                #print(reward)                    
                #reward_sum +=reward
                average_reward += reward_learning_rate * td_error.mean()
                
                #for x
                #average_current[==0] +=
                #print(average_reward)    
                #writer.add_scalar(f"Loss/scgf_{bias}",average_reward.mean(),i)
                #print("AVG",average_reward)
                
                #avg.append(average_reward.mean())
                #print(avg)
                x_state=next_state_x.clone()
                t_state=next_state_t
                wait_state = tau1
                i+=1
                rewards.append(reward)
                #avg_j += waiting_time.detach()*current
            
                if i%10==0:
                     target_net.load_state_dict(critic_model.state_dict())
                     #copy_actor.load_state_dict(actor_model.state_dict())
                if done:
                    #writer.flush()
                    
                    
                    #avg_j=eval(actor_model)
                    print("-------------DONE---------------------------")
                    break
               
           
       
        print('bias:{} SCGF:{}'.format(bias, average_reward.mean()))
        
       
        plt.scatter(bias,average_reward.mean().to("cpu").numpy())
        #plt.scatter(bias,(-average_reward.mean().to("cpu").numpy()))
        avg.append(average_reward.mean().to("cpu").numpy())
        #avg_current.append(avg_j[0])
        plt.savefig("avg")
        #scgf.append(np.array(total_mean_scgf).mean())
        time2=time.time()
        print('time:',time2-time1)
        
    avg_save=np.array(avg)
    out=0
    np.save(f"hypo_ratchet_2.npy",avg_save)
    np.save("currents_ratchet_from_1.npy",avg_current)
    #plt.figure()
    #plt.plot(biases,scgf_analytical,label='Analytical')
    plt.plot(biases,avg ,'.k',label='AC-RL')     
    plt.xlabel('s')
    plt.ylabel('scgf')
    plt.grid()
    plt.legend()
    plt.title(f"SCGF- Ratchet $J$, $p={p}$,$beta={q}$,$r={r}$") 
    plt.savefig(f"alpha_beta_{alpha}_{beta}.png")
if __name__ == '__main__':
    main()