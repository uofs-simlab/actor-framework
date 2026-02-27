#pragma once
#include <caf/all.hpp>

struct exit_actor_state {
        int completed = 0;
};

caf::behavior exit_actor_fun(caf::stateful_actor<exit_actor_state>* self,int limit); 
