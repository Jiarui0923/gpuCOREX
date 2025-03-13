class PropertyObject:
    def __getitem__(self, key): return self.__dict__[key] 
    def __setitem__(self, key, value): self.__dict__[key] = value
    def _load_dataframe(self, df):
         for key in df:
            self.__dict__[key] = df[key].values