from mlops.homework_03.utils.models.sklearn import train_model

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(model, encoded_data, all_data):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    train_data = encoded_data[0]
    dv = encoded_data[2]
    all_data = all_data['data_prep'][0]
    # Specify your transformation logic here
    model, metrics, y_pred = train_model(
        model,
        train_data,
        all_data['duration']
    )
    print(model.intercept_)
    return model, metrics, y_pred, dv
