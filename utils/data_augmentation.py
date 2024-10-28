"""
This script is an implementation suited for my needs of the paper EventDrop: Data Augmentation for Event-based learning (from F. Gu et al.)
https://github.com/fuqianggu/EventDrop

@GeoffroyK
"""
import random
import numpy as np

def event_drop(events_sequence: list, dims: tuple) -> list:
    """
    Apply event drop augmentation to an event sequence.

    This function implements the EventDrop augmentation technique described in
    "EventDrop: Data Augmentation for Event-based Learning" by F. Gu et al.
    It randomly selects one of four options:
    1. No drop (return original sequence)
    2. Drop by time
    3. Drop by area
    4. Random drop

    Args:
        events_sequence (list): A list of events, where each event is typically
                                represented as [timestamp, x, y, polarity].
        dims (tuple): A tuple of (width, height) representing the dimensions of the event frame.

    Returns:
        list: The augmented event sequence after applying the selected drop technique.

    Note:
        The specific drop techniques (drop_by_time, drop_by_area, random_drop) are
        implemented in separate functions.
    """
    option = np.random.randint(0,4)
    if option == 0: # No drop
        return events_sequence
    elif option == 1: # Drop by time
        return drop_by_time(events_sequence)
    elif option == 2: # Drop by area
        return drop_by_area(events_sequence, dims)
    else: # Random drop
        return random_drop(events_sequence)
    
def random_drop(events_sequence: list) -> list:
    """
    Randomly drop events from the input sequence.

    Args:
        events_sequence (list): A list of events, where each event is typically
                                represented as [timestamp, x, y, polarity].

    Returns:
        list: A new list of events with a random subset of events dropped.

    This function implements the "random drop" augmentation technique described in the
    EventDrop paper. It randomly selects a ratio between 10% and 90% of events to keep,
    and discards the rest.
    """
    ratio = np.random.randint(1,10)/10.
    # number of events in the sequence
    n_events = len(events_sequence)
    n_drop = int(n_events * ratio)
    idx = random.sample(list(np.arange(0,n_events)), n_events - n_drop)
    return events_sequence[idx]

def drop_by_area(events_sequence: list, dims: tuple) -> list:
    """
    Drop events within a randomly selected area of the event frame.    '''

    Args:
        events_sequence (numpy.ndarray): An array of shape (N, 4) where N is the number of events.
                                         Each row represents an event with [timestamp, x, y, polarity].
        dims (tuple): A tuple of (width, height) representing the dimensions of the event frame.

    Returns:
        numpy.ndarray: The events sequence with events dropped from the selected area.

    This function implements the "drop by area" augmentation technique described in the
    EventDrop paper. It randomly selects a rectangular area within the event frame and
    removes all events within that area. The area size is randomly chosen between 5% and 30%
    of the total frame area.
    """
    # x axis area selection
    x0  = np.random.uniform(dims[0])
    # y axis area selection
    y0 = np.random.uniform(dims[1])
    
    area_ratio = np.random.randint(1,6)/20.

    x_out = dims[0] * area_ratio
    y_out = dims[1] * area_ratio

    # Compute boundaries within dimension
    x0 = int(max(0, x0 - x_out/2.0))
    y0 = int(max(0, y0 - y_out/2.0))
    x1 = min(dims[0], x0 + x_out)
    y1 = min(dims[1], y0 + y_out)

    # Rectangle to be dropped
    x_drop = (x0, x1, y0, y1)

    idx1 = (events_sequence[:, 1] < x_drop[0]) | (events_sequence[:, 1] > x_drop[1])
    idx2 = (events_sequence[:, 2] < x_drop[2]) | (events_sequence[:, 2] > x_drop[3])
    idx = idx1 & idx2
    return events_sequence[idx]

def drop_by_time(events_sequence: list) -> list:
    '''
    Drop events within a randomly selected time window.

    Args:
        events_sequence (numpy.ndarray): An array of shape (N, 4) where N is the number of events.
                                         Each row represents an event with [timestamp, x, y, polarity].

    Returns:
        numpy.ndarray: The events sequence with events dropped from the selected time window.

    This function implements the "drop by time" augmentation technique described in the
    EventDrop paper. It randomly selects a time window and removes all events within that window.
    The time window start is uniformly sampled between 0 and 1, and its duration is randomly
    chosen between 0.1 and 1 of the total sequence duration.
    '''
    t_start = np.random.uniform(0,1)
    t_end = t_start + np.random.randint(1,10)/10.
    
    timestamps = events_sequence[:,0]
    max_t = max(timestamps)
    # Exclude events from the delimited time window.
    idx = (timestamps < max_t * t_start) | (timestamps > (max_t * t_end))
    return events_sequence(idx)