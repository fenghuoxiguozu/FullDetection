import weakref

class HookBase:
    def before_epoch(self):
        pass
      
    def before_iter(self):
        pass
      
    def after_epoch(self):
        pass
      
    def after_iter(self):
        pass
    
    
class Trainbase(HookBase):
    def __init__():
        self._hook = []
        
    def register(self,hooks):
        for h in self._hook:
            assert isinstance(hook,HookBase)
            h.trainer = weakref.proxy(self)
        self._hook.extend(hooks)
            
    def train(self,start_epoch,end_epoch):
        self.epoch = start_epoch
        
        self.before_epoch()
        for  in range(start_epoch,end_epoch)
            self.before_iter()
            self.run_sep()
            self.after_iter()
            self.epoch += 1
        self.after_epoch()
        
    def run_step():
        pass