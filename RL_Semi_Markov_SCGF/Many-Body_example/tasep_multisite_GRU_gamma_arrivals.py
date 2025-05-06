

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Exponential, Gamma, LogNormal,MixtureSameFamily
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import time
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()
##################################################Nets#################################################################

lattice_size=64 ## change lattice size, 2,10,64 were used in the paper
bond_dim = lattice_size+1
word_length = 1
batch_size=16
alpha=0.2
beta = 1


T=150000

if  torch.cuda.is_available():

    device = torch.device("cuda")
    print("here",torch.cuda.get_device_name(0))
    
else:
     device=torch.device("cpu")



### Neural Networks constructed with GRUs for many-body computations ###############
#class encoder()

class Actor(nn.Module):

      def __init__(self,):
          super(Actor,self).__init__()

          self.vect = 1
          self.inp_size=2
          self.hidden_encode= 16

          #self.encode           =   nn.LSTM(1,self.hidden_encode,batch_first=True)
          self.encode =nn.GRU(self.inp_size*word_length,self.hidden_encode,batch_first=True, bidirectional=True)
          self.encode2 =nn.GRU(2*self.hidden_encode,self.vect,batch_first=True)
          #self.encode2 =nn.LSTM(self.hidden_encode,64,batch_first=True)
          
          self.decode = nn.Linear(self.vect *int(np.floor(lattice_size/word_length)),128)
          #self.decode2 = nn.Linear(64,128)
          self.decode_move1=nn.Linear(128,lattice_size+1)
          #self.decode_move=nn.Linear(self.hidden_encode,lattice_size+1)


      def forward(self,x,constraints):
            out,h = self.encode(x)
            out,h = self.encode2(out)
            #out,h = self.encode2(out)
            #print(out.shape)
            code=out.reshape((batch_size, int(np.floor(lattice_size/word_length))*self.vect))
            #print(code.shape)
            decode = F.tanh(self.decode(code))
            #decode = F.relu(self.decode2(decode))
            move =torch.exp(self.decode_move1(decode)) * constraints
            move_probs=move/torch.sum(move,dim=1).unsqueeze(1)
            #print(move_probs.shape)
            return move_probs
#

class tau_net(nn.Module): # second actor for waiting_times

      def __init__(self,):
          super(tau_net,self).__init__()

          self.vect = 1
          self.hidden =3
          self.inp_size=2
          self.hidden_encode=16
          #self.encode           =   nn.LSTM(1,self.hidden_encode,batch_first=True)
          self.encode =nn.GRU(self.inp_size*word_length,self.hidden_encode,batch_first=True,bidirectional=True)
          self.encode2 =nn.GRU(2*self.hidden_encode,self.vect,batch_first=True)
          #self.encode2 =nn.LSTM(self.hidden_encode,64,batch_first=True)
          self.decode_time     = nn. Linear(self.vect *int(np.floor(lattice_size/word_length)),128)
          self.decode_concs      =  nn.Linear(128, self.hidden)
          self.decode_rates     =  nn.Linear(128, self.hidden)
          self.decode_weights     =  nn.Linear(128, self.hidden)

      def forward(self,x):
             out,h = self.encode(x)
             out,h = self.encode2(out)
             #print(out.shape)
             code=out.reshape((batch_size, int(np.floor(lattice_size/word_length))*self.vect))
             #code =F.tanh(self.encode(x))
             dist_primitives=F.tanh(self.decode_time(code))
             #dist_primitives=code
             concs = self.decode_concs(dist_primitives)
             rates  = self.decode_rates(dist_primitives)
             weights = self.decode_rates(dist_primitives)
             #concs= dist_primitives[:,:self.hidden]
             #rates= dist_primitives[:,self.hidden:2*self.hidden]
             #weights=dist_primitives[:,2 *self.hidden:3*self.hidden]
             return concs,rates,F.softmax(weights,dim=1)




class Critic(nn.Module):

      def __init__(self):
        super(Critic, self).__init__()
        #self.encode_state = nn.LSTM(1, 64)
        self.vect = 1
        self.inp_size=2
        self.hidden_encode1= 16
        self.encode =nn.GRU(self.inp_size * word_length,self.hidden_encode1,batch_first=True,bidirectional=True)
        self.encode2 =nn.GRU(2*self.hidden_encode1,self.vect,batch_first=True)
        #self.encode2 =nn.LSTM(self.hidden_encode1,64,batch_first=True)
        self.affine1=nn.Linear(self.vect *int(np.floor(lattice_size/word_length)),128)
        #self.affine2=nn.Linear(16,32)
        self.affine3=nn.Linear(128,1)
        #self.critic_head = nn.Conv1d(4, 1,1)
        self.state_value=[]


      def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x=F.relu(self.affine2(x))

        #x,_= self.encode_state(x)
        out,h = self.encode(x)
        #out,h = self.encode
        #print(out.shape)
        out,h = self.encode2(out)
        #print(out.shape)
        code=out.reshape((batch_size, int(np.floor(lattice_size/word_length))*self.vect))
        x = F.tanh(self.affine1(code))
        #x = F.relu(self.affine2(x))
        #state_value =self.affine2(x[:,-1,:])
        state_value =self.affine3(x)

        return state_value






def weights_init_uniform(m):
    classname=m.__class__.__name__
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                # Initialize weights with a normal distribution
                nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                # Initialize biases with zeros
                nn.init.zeros_(param)
    if classname.find('Linear')!=-1:
          torch.nn.init.xavier_uniform_(m.weight)
          m.bias.data.fill_(0.0001)
 
 
#def experience_replay_buffer():


policy_learning_rate=1e-4
value_learning_rate=1e-4
actor_model=Actor()
critic_model=Critic()
tau_model = tau_net()
actor_model.to(device)
critic_model.to(device)
tau_model.to(device)
target_net= Critic()
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
        
        
        
######################################################################################################Algorithm############################################

###exclusion constraints###
def asep_constraints(x): 
    #print("huh",x.shape)
    mask = torch.zeros(batch_size,lattice_size+1).to(device)
   
    #print(x[1,0,:],x[:,:,1])
    n=torch.zeros(batch_size,1).to(device)
    #if len(poss_moves)!=0:
    for i in range(batch_size):
        #print(x[i,:,0])
        poss_moves=torch.nonzero(x[i,:,0]) 
        mask2 = -1*torch.ones(lattice_size+1).to(device)
        #print(poss_moves)
        if len(poss_moves)!=0:
            poss_moves_next = 1+poss_moves.squeeze(1)
            #print(poss_moves_next)
            
            vals = torch.index_select(x[i,:,0], 0, poss_moves_next)
            #print("vals",vals,poss_moves_next,poss_moves)
            mask2.index_copy_(0,poss_moves_next, vals)
            #print("mask2",mask2)
            #mask2[poss_moves_next]= x[i,:,poss_moves_next]
            mask[i,mask2==0]=1
        if x[i,0,0]==0:
            mask[i,0]=1
        
            
    n[:,0] = torch.sum(mask[:,1:lattice_size],dim=1)
    #print("full", mask,x,n)
    return(mask,n)



####actions####
inp_length = 2
def get_action(x_state,t_mem, constraint_mask, bias):
    #x=torch.cat((x_state,t_state),dim=1)
    inp_length_jump=2   
    x=torch.cat((x_state,t_mem),dim=2)
    #print("x_before",x.shape)
    x=x.view(batch_size,-1,word_length*inp_length)
    #print("x_after",x.shape)
    #print(x_state.shape,t_mem.shape,x.shape)
    move_probs = actor_model(x,constraint_mask)
    m=Categorical(move_probs)
    d = m.sample()
    log_prob_trans = m.log_prob(d)
    actions=d.unsqueeze(1)
    total_log_prob= log_prob_trans.unsqueeze(1)
    #print(actions,"log_waiting and trans",log_prob_waiting.shape, log_prob_trans.unsqueeze(1).shape,waiting_time.unsqueeze(1).shape)
    return  actions, log_prob_trans.unsqueeze(1), total_log_prob


def get_tau(x_state,t_0):
      #x = torch.cat((x_state,t_0,q),dim=2)
      x = torch.cat((x_state,t_0),dim=2)
      x=x.view(batch_size,-1,word_length*inp_length)
      log_concs,log_rates,weights  = tau_model(x)
      concs,rates = torch.exp(log_concs),torch.exp(log_rates)
      mix = Categorical(weights)
      comp =Gamma(concs, rates)
      gmm = MixtureSameFamily(mix, comp)
      waiting_time=gmm.sample()
      log_prob_waiting=gmm.log_prob(waiting_time).unsqueeze(1)
      return waiting_time.unsqueeze(1), log_prob_waiting

###########values##################

def get_next_q(x_state,t_mem):
    #print(x_state.shape,tau_state.shape)
    x=torch.cat((x_state,t_mem),dim=2)
    x=x.view(batch_size,-1,word_length*inp_length)
    q_value=target_net(x)
    return q_value

def get_q_value(x_state,t_mem):
    x=torch.cat((x_state,t_mem),dim=2)
    x=x.view(batch_size,-1,word_length*inp_length)
    q_value=critic_model(x)
    return q_value
##############count_number_of_model_params###################

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



############################################################################################


initial_bias = 1
#bias	_step = 0.01
#bias_steps = 101
#biases = [initial_bias + bias_step*i for i in range(bias_steps)]
biases=np.arange(3.2,4.2,0.2)
    # bias loop
reward_learning_rate=5e-3


j= 0 
tau_0 = 1
  

num_params_actor1 = count(actor_model)
num_params_actor2 = count(tau_model)
num_params_critic = count(critic_model)
print("approx_num_params", num_params_actor1+num_params_actor2+num_params_critic)

##############################################Run Learning Loop################################################################################################################
def main():
    avg=[]    
    avg_current=[]
    J=[]
    for bias in biases:
        #batch_size=1
        # reset environment and episode reward
        reset_nets()
        x_state = torch.zeros((batch_size,lattice_size, 1),dtype=torch.float).to(device)
        t_state = torch.zeros((batch_size,1),dtype=torch.float).to(device)
        t_mem=torch.rand((batch_size,lattice_size, 1),dtype=torch.float).to(device)
        j_state= torch.zeros((batch_size,1),dtype=torch.float).to(device)
        wait_state = torch.rand((batch_size,1),dtype=torch.float).to(device)
        tau_0=torch.zeros(batch_size,1).to(device) 
        tau_01=torch.zeros(batch_size,1).to(device)     
        z=torch.zeros(batch_size,1,1).to(device)
        a_state=torch.cat((x_state,z),dim=1)
        constraint_mask,n=asep_constraints(x_state)
        #t_state_out=torch.zeros((batch_size,1),dtype=torch.float).to(device)
        #x_state[:,1]=1
        TD_error=[]
        total_mean_scgf=[]
        # long time loop while learning
        final_rate=0    
        epochs=1 ##change number of epochs for finite time trajectories
        #T=50000
        #batch_size=64
        #optimizer_actor.zero_grad()
        #optimizer_critic.zero_grad()
       
        for epoch in range(epochs):
            if epoch % 5 == 0:
               print(f"Epoch {epoch}")
            #reset_grads()
            time1=time.time()
            average_reward = 0
            s=0
            rewards=[]	
            values=[]
            targets=[]
            #replay_buffer=[]
            done=False	
	      	 #state = torch.zeros(batch_size, 1)
                  
            sim_time=torch.zeros(batch_size,1).to(device)
            #sim_time2=torch.zeros(batch_size,1).to(device)
            average_reward=torch.zeros(batch_size,1).to(device)
            #avg=[]
            t=[]
            itert=0
            #replay=memory()
            alpha=0.2
            beta=1
            b=1
            conc=2*torch.ones((batch_size,1)).to(device) 
            t_mem[:,0,0] = wait_state[:,0]
            while not done:
		    # select action from policy
		    #action_rates = select_action(state)
		    # take the actiong
                if itert%1000==0:
                   print(f"-------------------------------------------------sim_time:{sim_time.mean()},episode,{epoch},k,{bias},r, {average_reward.mean()}")
                
                #print(transition_prob)
                #################################################################################################

                
                action, log_prob_trans, total_log_prob=get_action(x_state,t_mem,constraint_mask,bias)
                next_state_x=x_state.clone()
                
                for k in range(batch_size):
                     if action[k,0]==0:
                        next_state_x[k,action[k,0],0] = 1
                        j_state[k,0] += 1
                         
                     elif action[k,0]==lattice_size:  
                          next_state_x[k,action[k,0]-1,0]=0
                     else:
                          next_state_x[k,action[k,0],0]  =1
                          next_state_x[k,action[k,0]-1,0]  =0 


            
                       

                z=torch.zeros(batch_size,1,1).to(device)
                a_state=torch.cat((next_state_x,z),dim=1)
                n_prev=n
                constraint_mask,n=asep_constraints(a_state)
                #################################################################################################
                
                M1=torch.zeros((batch_size,1)).to(device)   
                S1=torch.zeros((batch_size,1)).to(device)
                n_0 = x_state[:,0,0].unsqueeze(1)
                n_L = x_state[:,-1,0].unsqueeze(1)
                
                
                
                t_mem_prev = t_mem.clone()
                mask2=torch.zeros((batch_size,1)).to(device) 
                #survival1=torch.special.gammaincc(conc,alpha*(wait_state+tau_0))
                #print(tau_0,tau10,tau00)
                #print(x_state.squeeze(),next_state_x.squeeze(),tau_0)
                prob_wtd1= (wait_state+tau_0)*torch.exp(-(wait_state+tau_0)*alpha) * alpha**2
        
                
                M1 =  prob_wtd1/torch.special.gammaincc(conc,alpha*(wait_state+tau_0))
                #print("M1",M3.shape)
                #print(n_0,n_prev,n_L)
                den= M1*(1-n_0)+(n_prev)+(beta*n_L)
                mask2[action[:,0]==0]=M1[action[:,0]==0]
                mask2[action[:,0]!=0]= 1     
                mask2[action[:,0]==beta]=beta
            
                mask2=mask2/den
                #print(mask2)
                
                #print(tau_0[:,0])
                
                
                n_01 = next_state_x[:,0,0].unsqueeze(1)
                n_L1 = next_state_x[:,-1,0].unsqueeze(1)

                tau_0[action[:,0]==0]= 0   #[0,0]
                tau_0[action[:,0]!=0] += wait_state[action[:,0]!=0]
               # print(tau_0)
                tau_0[n_0[:,0]==1]=0
                #tau_0[:,0][n_0[:,0]==1]=0
                for u in range(lattice_size):
                   t_mem[:,u,0]=wait_state[:,0] 
          
                t_mem[:,0,0] = tau_0[:,0]
                 
                waiting_time, log_prob_waiting= get_tau(next_state_x,t_mem)
                tau1 = waiting_time.detach()

                tau_01[:,0] = tau1[:,0] + tau_0[:,0]
                tau_01[n_01[:,0]==1]= 0
                for u in range (lattice_size):
                   t_mem[:,u,0]=tau1[:,0] 
                t_mem[:,0,0]=tau_01[:,0] 


                #print(x_state.squeeze(), next_state_x.squeeze(),tau_0,action, wait_state+tau_0)
            
                                                                           
                bias_tensor= torch.ones((batch_size,1)).to(device)
                log_org_waiting1=torch.zeros((batch_size,1)).to(device)
                log_org_waiting2=torch.zeros((batch_size,1)).to(device)
                #log_org_pro = torch.zeros((batch_size,1)).to(device)
                #
                survival1=torch.special.gammaincc(conc,alpha*(tau1+tau_0)) 
                  
                
                prob_wtd1= (tau1+tau_0)*torch.exp(-(tau1+tau_0)*alpha) * alpha**2
                #M1[tau_0[:,0]==0]= prob_wtd[tau_0[:,0]==0]
                M1 =  prob_wtd1/torch.special.gammaincc(conc,alpha*(tau_0+tau1))
        

                #M1[n_01[:,0]==1]=0
                #print(tau_0.shape,S1.shape,survival1.shape)
                S1[tau_0[:,0]==0] = torch.log(survival1[tau_0[:,0]==0])
                #print("s1",S1.shape)
                S1[tau_0[:,0]!=0] = torch.log(survival1[tau_0[:,0]!=0]) - torch.log(torch.special.gammaincc(conc[tau_0[:,0]!=0],alpha*(tau_0[tau_0[:,0]!=0])))
            
                #print(next_state_x.squeeze(),n_01,n,n_L1)
                log_prob_org_waiting_1 = torch.log(M1*(1-n_01) + (n) + (beta* n_L1)) 
                log_prob_org_waiting_2 =   S1*(1-n_01) - ((n+ n_L1* beta)*tau1)
                #print("tau1",tau1,"s1",S1,"s2",S2,"s3",S3)
            
                
                
                bias_tensor[action[:,0]!=0]=0
                
                I = log_prob_waiting.detach()-(log_prob_org_waiting_1+log_prob_org_waiting_2) + log_prob_trans.detach() - torch.log(mask2)
                
                reward=(bias*bias_tensor -I)
                
                
                next_state_t= t_state+waiting_time
               
                #tau_0[action[:,0]==lattice_size] += tau1[action[:,0]==lattice_size]
                #tau_0[n_01[:,0]==1]=0
                #print("s0",x_state,"s1",next_state_x)
                sim_time += waiting_time
                #print(sim_time)
                t.append(sim_time.mean())
                if sim_time.mean()>=T:
                   done= True
                
               
                       
                with torch.no_grad():
                     #next_waiting_time, _, _ ,_ ,_= get_action(next_state_t,next_state_x)
                     next_value=get_next_q(next_state_x,t_mem)
               
                value=get_q_value(x_state,t_mem_prev)
              
                target=((next_value)+ (reward) - ((tau1)*average_reward)) 
                
         
         
              
                td_error=(target-value.detach())
                #print(td_error.mean())
                TD_error.append(td_error.mean())
                policy_loss = torch.mean(- (total_log_prob) * (td_error.mean()))
                policy_loss2 = torch.mean(- (log_prob_waiting) * (td_error.mean()))
                #print(tau_alpha)
                critic_loss=F.mse_loss(value,target)
                loss=policy_loss.item()
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                optimizer_tau.zero_grad()
                
             
                loss_value=critic_loss.item()   
                policy_loss.backward()
                policy_loss2.backward()
                critic_loss.backward()       
                optimizer_actor.step()
                optimizer_critic.step()
                optimizer_tau.step()
                
                
             
                
                average_reward += reward_learning_rate * td_error.mean()
                
         
                x_state=next_state_x.clone()
                t_state=next_state_t    
                wait_state = tau1
                itert+=1
                rewards.append(reward)
                if itert%10==0:
                     target_net.load_state_dict(critic_model.state_dict())
                if done:
                    #writer.flush()
                   
                    print("-------------DONE---------------------------")
                    break
               
           
       
        print('bias:{} SCGF_current:{}'.format(bias, average_reward.mean()))
        time2=time.time()
        print('time:',time2-time1)
        J.append((j_state.mean()/t_state.mean()).to("cpu").numpy())
        #plt.scatter(bias,average_reward.mean().to("cpu").numpy())
        #plt.scatter(bias,(-average_reward.mean().to("cpu").numpy()))
        avg.append(average_reward.mean().to("cpu").numpy())
        avg_current.append(s/sim_time)
        np.save("multi_gamma_tasep_118_2.npy",np.array(avg)) #save_scgf
        np.save("gamma_current_118_2.npy",np.array(J)) #save_current
        plt.savefig("avg")
        #scgf.append(np.array(total_mean_scgf).mean())
    avg_save=np.array(avg)
    out=1
    a=2
    np.save("multi_gamma_tasep_118_2.npy",avg_save) #save scgf
    np.save("gamma_current_118_2.npy",np.array(J)) # save current
    plt.figure()
    plt.plot(biases,scgf_analytical,label='Analytical')
    plt.plot(biases,avg ,'.g',label='AC-RL')     
    plt.xlabel('s')
    plt.ylabel('scgf')
    plt.grid()
    plt.legend()
    plt.title(f"SCGF- ASEP_one_Site $J_{out}$ alpha={alpha},beta={beta}, a={a}") 
    plt.savefig(f"alpha_beta_{alpha}_{beta}.png")

if __name__ == '__main__':
    main()
