import torch

class GaussianBatchNormalizer(object):
    def __init__(self, eps=1e-5):
        super(GaussianBatchNormalizer, self).__init__()
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.n = 0  # Anzahl der Samples

    def update(self, x):
        """Aktualisiere Mittelwert und Varianz mit neuen Batch-Daten."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)  # Bessere Schätzung für Varianz

        # Berechne den neuen Mittelwert und die neue Varianz
        new_n = self.n + x.size(0)
        self.mean = (self.n * self.mean + x.size(0) * batch_mean) / new_n
        self.var = (self.n * self.var + x.size(0) * batch_var) / new_n
        
        self.n = new_n

    def encode(self, x):
        """Normalisiere die Eingabedaten."""
        if self.n == 0:  # Wenn noch keine Daten aktualisiert wurden
            raise ValueError("Update the normalizer with data before encoding.")
        
        x = (x - self.mean) / (torch.sqrt(self.var + self.eps))
        return x

    def decode(self, x):
        """Dekodiert die normalisierten Daten zurück zu den ursprünglichen Werten."""
        if self.n == 0:  # Wenn noch keine Daten aktualisiert wurden
            raise ValueError("Update the normalizer with data before decoding.")
        
        x = (x * torch.sqrt(self.var + self.eps)) + self.mean
        return x
