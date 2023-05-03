appName="azure-sql-db-python-rest-api"
resourceGroup="my-resource-group"

az webapp config connection-string set \
    -g $resourceGroup \
    -n $appName \
    --settings WWIF=$SQLAZURECONNSTR_WWIF \
    --connection-string-type=SQLAzure