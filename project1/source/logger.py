#!/usr/bin/env python3
import neptune

class BaseLogger:
    """ This is base class of all loggers.
    It makes it easier to run experiments without logging"""
    def __init__(self, tags, args):
        pass

    @classmethod
    def new_experiment(cls, tags, args):
        return cls(tags, args)

    # def log_hyperparams(self, hyperparams):
        # """ Set hyperparameters of experiment. """
        # pass

    def log_status(self, text):
        pass

    def log_metric(self, metric, value):
        pass

    def log_metrics(self, dictionary):
        pass
    def stop(self):
        pass
    def log_text(self, name, text):
        pass

class NeptuneLogger(BaseLogger):
    def __init__(self, tags, args):
        neptune.set_project('pixelneo/retrieval')
        neptune.create_experiment(params=args._data)
        for tag in tags:
            neptune.append_tag(tag)

    @classmethod
    def new_experiment(cls, tags, args):
        return cls(tags, args)

    # def log_hyperparams(self, hyperparams):
        # """ Set hyperparameters of experiment. """
        # for k, v in hyperparams._data.items():
            # if isinstance(v, list):
                # neptune.set_property(k, '; '.join(v))
            # else:
                # neptune.set_property(k, v)
    def log_status(self, text):
        neptune.log_text('status', text)

    def log_text(self, name, text):
        neptune.log_text(name, text)

    def log_metric(self, metric, value):
        neptune.log_metric(metric, value)

    def log_metrics(self, dictionary):
        for k,v in dictionary.items():
            self.log_metric(k,v)

    def stop(self):
        neptune.stop()
